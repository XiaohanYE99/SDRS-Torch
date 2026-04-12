"""Train PD references: sine bank or MLP(θ, θ̇) -> hinge accel -> PD target."""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot import Robot
from simulator import ConvexHullPBADSimulator, SimStepFn

# Gains: simulator PD only acts on dof>=6 (hinges)
DT, KP, KD = 0.01, 1000.0, 100.0
HORIZON, EPOCHS, LR = 80, 50, 1e-2
N_SIN = 4
FRAME_DT = 0.03
N_ROOT = 6
MLP_HIDDEN = 64
DEFAULT_TRAJ_PATH = 'train_traj_final.pt'
DEFAULT_LOGDIR = 'runs/train'


class PdSineBank(nn.Module):
    """
    **Position** reference on hinge dofs (simulator ``pd_target`` = p):

        p(t) = anchor + sum_k B_k * sin(C_k * t + φ_k)

    **Velocity** (planning): dp/dt = sum_k B_k * C_k * cos(C_k * t + φ_k).
    """

    def __init__(self, robot: Robot, n_sin: int, dt: float, base: torch.Tensor,
                 n_root: int = N_ROOT):
        super().__init__()
        dev, dty = base.device, base.dtype
        n_dof = robot.n_dof
        n_hinge = max(0, n_dof - n_root)
        self.n_dof = n_dof
        self.n_root = n_root
        self.n_sin = n_sin
        self.dt = dt
        self.register_buffer('anchor', base.clone().detach())
        lo_list, hi_list = [], []
        for j in robot.joints:
            if j.jtype == 'hinge':
                lo_list.append(j.limit_lower + 0.05)
                hi_list.append(j.limit_upper - 0.05)
        assert len(lo_list) == n_hinge, (
            f'hinge count {len(lo_list)} vs n_dof-n_root={n_hinge}')
        if n_hinge:
            self.register_buffer(
                '_hinge_lo',
                torch.tensor(lo_list, device=dev, dtype=dty).view(1, -1))
            self.register_buffer(
                '_hinge_hi',
                torch.tensor(hi_list, device=dev, dtype=dty).view(1, -1))
        else:
            self.register_buffer('_hinge_lo', torch.zeros(0, device=dev, dtype=dty))
            self.register_buffer('_hinge_hi', torch.zeros(0, device=dev, dtype=dty))
        self.B = nn.Parameter(torch.zeros(n_hinge, n_sin, device=dev, dtype=dty))
        self.C = nn.Parameter(torch.full((n_hinge, n_sin), 3.0, device=dev, dtype=dty))
        self.phase = nn.Parameter(torch.zeros(n_hinge, n_sin, device=dev, dtype=dty))
        with torch.no_grad():
            self.B.uniform_(0.01, 0.04)
            self.C.add_(torch.randn_like(self.C) * 0.25)
            self.C.clamp_(min=1.2, max=7.0)
            self.phase.uniform_(-3.14159, 3.14159)

    def _hinge_wave_and_dot(
            self, n_steps: int
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        dev, dty = self.anchor.device, self.anchor.dtype
        n_h = self.B.shape[0]
        if n_h == 0:
            z = torch.zeros(0, device=dev, dtype=dty)
            return z, z
        t = torch.arange(n_steps, device=dev, dtype=dty) * self.dt
        t = t.view(-1, 1, 1).expand(-1, n_h, self.n_sin)
        b = self.B.unsqueeze(0)
        c = self.C.unsqueeze(0)
        ph = self.phase.unsqueeze(0)
        ang = c * t + ph
        wave = (b * torch.sin(ang)).sum(dim=-1)
        wave_dot = (b * c * torch.cos(ang)).sum(dim=-1)
        return wave, wave_dot

    def pd_dot_at_steps(self, n_steps: int) -> torch.Tensor:
        dev, dty = self.anchor.device, self.anchor.dtype
        n_h = self.B.shape[0]
        out = torch.zeros(n_steps, self.n_dof, device=dev, dtype=dty)
        if n_h == 0:
            return out
        _, wave_dot = self._hinge_wave_and_dot(n_steps)
        out[:, self.n_root:self.n_root + n_h] = wave_dot
        return out

    def pd_at_steps(self, n_steps: int) -> torch.Tensor:
        dev, dty = self.anchor.device, self.anchor.dtype
        n_h = self.B.shape[0]
        if n_h == 0:
            return self.anchor.unsqueeze(0).expand(n_steps, -1)
        wave_h, _ = self._hinge_wave_and_dot(n_steps)
        anchor_h = self.anchor[self.n_root:self.n_root + n_h].view(1, -1)
        hinge_pd = anchor_h + wave_h
        hinge_pd = torch.clamp(hinge_pd, self._hinge_lo, self._hinge_hi)
        out = self.anchor.unsqueeze(0).expand(n_steps, -1).clone()
        out[:, self.n_root:self.n_root + n_h] = hinge_pd
        return out


class PdAccelMLP(nn.Module):
    """
    每步用当前帧广义坐标 ``θ_t`` 与离散速度 ``θ̇_t ≈ (θ_t - θ_{t-1})/dt`` 拼成输入
    ``[θ_t ; θ̇_t]``（维数 ``2 n_dof``），MLP 输出**铰链**加速度 ``a``（维数 ``n_hinge``）。

    PD 位置目标（全向量 ``pd``）：

    - 根自由度 ``pd[:n_root] = θ_t[:n_root]``（与仿真 ``joint_mask`` 一致，根上无 PD 弹力）；
    - 铰链： ``pd_h = θ_h + θ̇_h dt + ½ a dt²``，再按关节限幅做 clamp。
    """

    def __init__(
            self, robot: Robot, dt: float, base: torch.Tensor,
            n_root: int = N_ROOT, hidden: int = MLP_HIDDEN):
        super().__init__()
        dev, dty = base.device, base.dtype
        n_dof = robot.n_dof
        n_hinge = max(0, n_dof - n_root)
        self.n_dof = n_dof
        self.n_root = n_root
        self.n_hinge = n_hinge
        self.dt = float(dt)
        self.register_buffer('anchor', base.clone().detach())
        in_dim = 2 * n_dof
        lo_list, hi_list = [], []
        for j in robot.joints:
            if j.jtype == 'hinge':
                lo_list.append(j.limit_lower + 0.05)
                hi_list.append(j.limit_upper - 0.05)
        assert len(lo_list) == n_hinge
        if n_hinge:
            self.register_buffer(
                '_hinge_lo',
                torch.tensor(lo_list, device=dev, dtype=dty).view(1, -1))
            self.register_buffer(
                '_hinge_hi',
                torch.tensor(hi_list, device=dev, dtype=dty).view(1, -1))
        else:
            self.register_buffer('_hinge_lo', torch.zeros(0, device=dev, dtype=dty))
            self.register_buffer('_hinge_hi', torch.zeros(0, device=dev, dtype=dty))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, device=dev, dtype=dty),
            nn.Tanh(),
            nn.Linear(hidden, hidden, device=dev, dtype=dty),
            nn.Tanh(),
            nn.Linear(hidden, n_hinge, device=dev, dtype=dty),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def hinge_accel_and_pd(self, th: torch.Tensor, thm: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``θ̇ = (th - thm)/dt``，MLP → ``a``；``pd`` 为全维 PD 目标。"""
        dt = self.dt
        qdot = (th - thm) / dt
        inp = torch.cat([th, qdot], dim=0)
        a = self.net(inp)
        pd = th.clone()
        if self.n_hinge == 0:
            return a, pd
        sl = slice(self.n_root, self.n_root + self.n_hinge)
        qh = th[sl]
        qdh = qdot[sl]
        qh_tgt = qh + qdh * dt + 0.5 * a * (dt * dt)
        lo = self._hinge_lo.view(-1)
        hi = self._hinge_hi.view(-1)
        qh_tgt = torch.max(torch.min(qh_tgt, hi), lo)
        pd[sl] = qh_tgt
        return a, pd


def rollout_trajectory(
        robot, sim, base, horizon, kp, kd, *,
        mode: str = 'sin',
        bank: Optional[PdSineBank] = None,
        mlp: Optional[PdAccelMLP] = None,
        ) -> List[torch.Tensor]:
    """CPU list of theta; ``mode`` is ``'sin'`` or ``'mlp'``."""
    with torch.no_grad():
        sim.reset_episode_state()
        traj: List[torch.Tensor] = [base.cpu().clone()]
        th, thm = base.clone(), base.clone()
        if mode == 'sin':
            assert bank is not None
            pd_seq = bank.pd_at_steps(horizon)
            for t in range(horizon):
                thp1, _, _ = sim.step(th, thm, pd_seq[t], kp, kd)
                traj.append(thp1.cpu().clone())
                thm, th = th, thp1
        else:
            assert mlp is not None
            for _ in range(horizon):
                _, pd = mlp.hinge_accel_and_pd(th, thm)
                thp1, _, _ = sim.step(th, thm, pd, kp, kd)
                traj.append(thp1.cpu().clone())
                thm, th = th, thp1
    return traj


def save_trajectory(
        path: str, traj: List[torch.Tensor], meta: Dict[str, Any]) -> None:
    theta = torch.stack([q if isinstance(q, torch.Tensor) else torch.tensor(q)
                         for q in traj], dim=0)
    payload = {'theta': theta, 'meta': meta}
    torch.save(payload, path)
    print(f"[train] saved trajectory ({theta.shape[0]} frames) -> {path}")


def load_trajectory(path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location='cpu')
    return payload['theta'], payload.get('meta', {})


def playback_traj(robot, traj, *, frame_dt=FRAME_DT, text_only=False):
    dev, dty = robot.device, robot.dtype
    if not text_only:
        try:
            from visualizer import SpiderVisualizer
            viz = SpiderVisualizer(robot)
            th0 = traj[0].to(device=dev, dtype=dty)
            plotter, link_data = viz.open_plotter(
                th0, title='train trajectory')
            for i, q in enumerate(traj):
                th = q.to(device=dev, dtype=dty)
                bp = th[:3].detach().cpu().numpy()
                txt = (f"step {i}/{len(traj) - 1}  "
                       f"body ({bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.3f})")
                if not viz.update_frame(plotter, link_data, th, txt):
                    break
                time.sleep(frame_dt)
            plotter.close()
            return
        except ImportError:
            print("[train] pyvista not installed; falling back to text + sleep.")
    for i, q in enumerate(traj):
        print(f"step {i:3d}  theta={q.tolist()}")
        if i < len(traj) - 1:
            time.sleep(frame_dt)


def _parse_args():
    p = argparse.ArgumentParser(description='PBAD train / render trajectory')
    p.add_argument('--xml', type=str, default='ant.xml',
                   help='Robot XML under python_pbad/ (default: ant.xml)')
    p.add_argument('--pd', choices=('sin', 'mlp'), default='sin',
                   help='PD source: sine bank or MLP(accel integration)')
    p.add_argument('--render-traj', type=str, default=None, metavar='PATH',
                   help='Load saved trajectory and render only (no training)')
    p.add_argument('--traj-out', type=str, default=DEFAULT_TRAJ_PATH,
                   help='Where to save final trajectory after training')
    p.add_argument('--logdir', type=str, default=DEFAULT_LOGDIR,
                   help='TensorBoard log directory')
    p.add_argument('--no-tb', action='store_true', help='Disable TensorBoard')
    p.add_argument('--text', action='store_true', help='Text playback, no PyVista')
    return p.parse_args()


def _resolve_xml(script_dir: str, name: str) -> str:
    return name if os.path.isabs(name) else os.path.join(script_dir, name)


def main():
    args = _parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dty = torch.float64

    if args.render_traj:
        traj_tensor, meta = load_trajectory(args.render_traj)
        traj = [traj_tensor[i].clone() for i in range(traj_tensor.shape[0])]
        xml_path = meta.get('xml')
        if not xml_path or not os.path.isfile(xml_path):
            xml_path = _resolve_xml(script_dir, args.xml)
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f'Robot XML not found: {xml_path}')
        robot = Robot(xml_path, device=dev, dtype=dty)
        print(f"[train] loaded {len(traj)} frames from {args.render_traj}  xml={xml_path}  meta={meta}")
        playback_traj(robot, traj, frame_dt=FRAME_DT, text_only=args.text)
        return

    xml = _resolve_xml(script_dir, args.xml)
    if not os.path.isfile(xml):
        raise FileNotFoundError(f'Robot XML not found: {xml}')
    robot = Robot(xml, device=dev, dtype=dty)
    sim = ConvexHullPBADSimulator(robot, dt=DT, device=dev, _output=False)
    base = robot.default_theta().clone()

    writer = None
    if not args.no_tb:
        try:
            from torch.utils.tensorboard import SummaryWriter
            logdir = args.logdir
            if not os.path.isabs(logdir):
                logdir = os.path.join(script_dir, logdir)
            os.makedirs(logdir, exist_ok=True)
            writer = SummaryWriter(log_dir=logdir)
            print(f"[train] TensorBoard log dir: {logdir}")
        except ImportError:
            print('[train] tensorboard not installed; use `pip install tensorboard`')

    if args.pd == 'sin':
        policy: nn.Module = PdSineBank(robot, N_SIN, DT, base)
    else:
        policy = PdAccelMLP(robot, DT, base)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    for ep in range(EPOCHS):
        sim.reset_episode_state()
        th = base.clone()
        thm = th.clone()
        x_series = [th[0].clone()]
        if args.pd == 'sin':
            bank = policy
            pd_seq = bank.pd_at_steps(HORIZON)
            for t in range(HORIZON):
                tp1 = SimStepFn.apply(pd_seq[t], th, thm, None, sim, KP, KD)
                thm, th = th, tp1
                x_series.append(th[0].clone())
        else:
            mlp = policy
            for _ in range(HORIZON):
                _, pd = mlp.hinge_accel_and_pd(th, thm)
                tp1 = SimStepFn.apply(pd, th, thm, None, sim, KP, KD)
                thm, th = th, tp1
                x_series.append(th[0].clone())
        x_traj = torch.stack(x_series)
        vx = (x_traj[1:] - x_traj[:-1]) / DT
        v = vx.mean()
        # v>0: loss=-v^2（鼓励正向速度）；v≤0: loss=v^2（惩罚反向或静止）
        loss = torch.where(v > 0, -(v * v), v * v)*1e6
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if writer is not None:
            writer.add_scalar('loss', loss.item(), ep)
            writer.add_scalar('mean_vx', float(v), ep)
            writer.add_scalar('dx_body_x', float(x_traj[-1] - x_traj[0]), ep)
        if (ep + 1) % 1 == 0 or ep == 0:
            dx = float(x_traj[-1] - x_traj[0])
            print(f"epoch {ep + 1}/{EPOCHS}  loss={loss.item():.4f}  "
                  f"mean_vx={float(v):.4f} m/s  dx={dx:.4f} m")

    if writer is not None:
        writer.flush()
        writer.close()

    traj_out = args.traj_out
    if not os.path.isabs(traj_out):
        traj_out = os.path.join(script_dir, traj_out)

    print("\n--- replay optimized trajectory ---")
    if args.pd == 'sin':
        traj = rollout_trajectory(
            robot, sim, base, HORIZON, KP, KD, mode='sin', bank=policy)
    else:
        traj = rollout_trajectory(
            robot, sim, base, HORIZON, KP, KD, mode='mlp', mlp=policy)

    meta = {
        'pd_mode': args.pd,
        'mlp_input': 'concat_theta_theta_dot' if args.pd == 'mlp' else None,
        'mlp_in_dim': 2 * robot.n_dof if args.pd == 'mlp' else None,
        'horizon': HORIZON,
        'dt': DT,
        'kp': KP,
        'kd': KD,
        'xml': xml,
        'epochs': EPOCHS,
    }
    save_trajectory(traj_out, traj, meta)
    playback_traj(robot, traj, frame_dt=FRAME_DT, text_only=args.text)


if __name__ == '__main__':
    main()
