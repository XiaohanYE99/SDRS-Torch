"""Entry: debug / batch_debug / run_sim / batch_run_sim. Run with no args for usage.

debug-mode flags:
  --energy-only   Only ``debug_energy`` (DDE-* directional Hessian checks).
  --xl            Use ``simulator_adapt`` and add the DTDXL ANA/AD/FD compare.
  --xl-only       Like --xl but skip the regular three sections.
  --gtol/--delta/--trials/--head/--seed/--no-autograd : XL tuning.

Top-level aliases:
  debug_energy  ==  debug --energy-only
  debug_xl      ==  debug --xl-only
"""

import math
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Repo root, so we can import the standalone XL-debug helper module.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from robot import Robot
from simulator import ConvexHullPBADSimulator

try:
    from batch_simulator import BatchSimulator
    HAS_BATCH = True
except ImportError:
    HAS_BATCH = False

DTYPE = torch.float64


# ---------------------------------------------------------------------------
#  XL-debug helpers (lazy-imported; only needed when the user passes --xl /
#  --xl-only / runs the ``debug_xl`` mode, OR --energy-only --xl).
# ---------------------------------------------------------------------------
def _import_xl_helpers():
    """Return ``(AdaptSim, run_dtdxl_compare)`` for vertex-column gradient
    debugging.  Raises a clear ImportError if either component is missing."""
    try:
        from simulator_adapt import (
            ConvexHullPBADSimulator as AdaptSim)  # noqa: WPS433
    except ImportError as ex:
        raise ImportError(
            "simulator_adapt is required for --xl / debug_xl but was not "
            "found on PYTHONPATH.") from ex
    try:
        from debug_adapt_vertex import run_dtdxl_compare  # noqa: WPS433
    except ImportError as ex:
        raise ImportError(
            "debug_adapt_vertex.py (in repo root) provides "
            "run_dtdxl_compare; it could not be imported.") from ex
    return AdaptSim, run_dtdxl_compare


def _default_pd_seq(robot, n_steps, device, dtype):
    """[T, n] constant PD reference (matches ``default_theta()`` incl. root height)."""
    p = robot.default_theta().to(device=device, dtype=dtype)
    return p.unsqueeze(0).expand(n_steps, -1).clone()


def _default_pd_seq_batch(robot, N, n_steps, device, dtype, dt: float,
                          joint_amp: float = 0.22, gait_freq: float = 1.5):
    """[T, N, n] PD reference: body height + **non-zero joint targets** (sin over time).

    ``default_theta()`` is mostly zeros on hinge DOFs, so a constant expand looks
    like "PD=0" on joints.  Here dof>=6 get ``base + joint_amp * sin(2π f t + φ)``,
    with independent phase ``φ`` per (env, joint).
    """
    n = robot.n_dof
    base = robot.default_theta().to(device=device, dtype=dtype).clone()
    pd_seq = base.unsqueeze(0).unsqueeze(0).expand(n_steps, N, n).clone()
    if n <= 6:
        return pd_seq
    t = torch.arange(n_steps, device=device, dtype=dtype).view(n_steps, 1, 1) * float(dt)
    phase = torch.rand(N, n - 6, device=device, dtype=dtype) * (2.0 * math.pi)
    w = 2.0 * math.pi * float(gait_freq)
    j0 = base[6:].view(1, 1, -1).expand(n_steps, N, -1)
    pd_seq[:, :, 6:] = j0 + float(joint_amp) * torch.sin(w * t + phase.unsqueeze(0))
    return pd_seq


def _batch_initial_theta_spread_root_y(
        robot, N: int, device, dtype, *,
        ymax_extra: float = 0.14, ground_clearance: float = 0.018):
    """``[N, n_dof]``: same as ``default_theta()`` except root world-Y (DOF index 1)
    linearly spaced per env, strictly above the ground mesh (FK + AABB).

    Spider / Y-up: free-joint translation Y is index ``1``."""
    base = robot.default_theta().to(device=device, dtype=dtype).clone()
    n = robot.n_dof
    th = base.unsqueeze(0).expand(N, n).clone()
    iy = 1
    by = float(base[iy].item())
    if robot.ground is not None:
        with torch.no_grad():
            wv, mask = robot.get_wv_stacked_batch(base.unsqueeze(0))
            wv0 = wv[0]
            low_y = float(wv0[:, :, iy][mask].min().item()) if mask.any() else by - 0.25
            g_top = float(robot.ground.vertices[:, iy].max().item())
            # Lowest hull vertex rises ~1:1 with root y; keep min vertex above ground.
            y_min = g_top + ground_clearance + (by - low_y)
    else:
        y_min = by + 0.02
    y_max = by + ymax_extra
    if y_max <= y_min:
        y_max = y_min + 0.05
    th[:, iy] = torch.linspace(y_min, y_max, N, device=device, dtype=dtype)
    return th


# =====================================================================
#  1. debug — single-env, runs all debug checks
# =====================================================================

def debug(xml_path, device='cpu', *, use_random=False, scale=0.1,
          warmup_steps=30, render=False,
          # selectors
          energy_only: bool = False,
          xl: bool = False,
          xl_only: bool = False,
          # XL tuning
          xl_gtol: float = 1e-7,
          xl_max_iter: int = 2000,
          xl_delta: float = 1e-6,
          xl_trials: int = 3,
          xl_head: int = 8,
          xl_seed: int = 0,
          xl_no_autograd: bool = False):
    """Single-env debug.

    Default
        Uses :mod:`simulator` and runs ``debug_energy`` (DDE-* directional
        Hessian checks) + ``debug_backward`` (IFT) + ``debug_verify_theta_derivatives_fd``
        (θ gradient/Hessian FD).

    --energy-only
        Only the ``debug_energy`` section (every ``DDE-*`` line: DDE / DDE-L /
        DDE-LL / DDE-P / DDE-XL with ANA / FD / AD triples).  Picks
        :mod:`simulator_adapt` automatically when combined with ``--xl``.

    --xl / --xl-only
        Use :mod:`simulator_adapt` and additionally run
        :func:`debug_adapt_vertex.run_dtdxl_compare` (the detailed DTDXL
        ANA / Autograd / FD comparison).  ``--xl-only`` skips the regular
        three sections.

    Combos
        ``--energy-only --xl``  ⇒  only DDE-* checks, but using the **adapted**
        simulator (so DDE-XL goes through the adapted ``_compute_HThetaD`` /
        ``_jvp_grad_theta_wrt_local_verts_autograd``).
    """
    use_adapt = bool(xl or xl_only)

    # --- which sections will run (resolved from selectors) ----------------
    run_energy = not xl_only
    run_backward = (not energy_only) and (not xl_only)
    run_theta_fd = (not energy_only) and (not xl_only)
    run_xl_compare = use_adapt and (not energy_only)

    title_bits = []
    if run_energy:
        title_bits.append("energy")
    if run_backward:
        title_bits.append("backward")
    if run_theta_fd:
        title_bits.append("θ-FD")
    if run_xl_compare:
        title_bits.append("XL")
    if not title_bits:
        title_bits = ["(nothing selected)"]

    print("=" * 60)
    if energy_only and not use_adapt:
        print("  ConvexHullPBAD Energy Debug (DDE-* only, simulator)")
    elif energy_only and use_adapt:
        print("  ConvexHullPBAD Energy Debug (DDE-* only, simulator_adapt)")
    elif xl_only:
        print("  ConvexHullPBAD Debug — XL ONLY (simulator_adapt)")
    elif use_adapt:
        print("  ConvexHullPBAD Full Debug (energy + backward + θ-FD + XL, simulator_adapt)")
    else:
        print("  ConvexHullPBAD Full Debug (energy + backward + θ-FD)")
    print(f"  sections: {' + '.join(title_bits)}")
    print("=" * 60)

    robot = Robot(xml_path, device=device, dtype=DTYPE)
    if use_adapt:
        AdaptSim, run_dtdxl_compare = _import_xl_helpers()
        # XL-only / energy-only: keep step()'s LM logs quiet.
        sim_output = not (xl_only or energy_only)
        sim = AdaptSim(
            robot, dt=0.01, device=device, _output=sim_output,
            gtol=xl_gtol, max_newton_iter=xl_max_iter)
    else:
        sim_output = not energy_only
        sim = ConvexHullPBADSimulator(
            robot, dt=0.01, device=device, _output=sim_output)

    theta = theta_t = theta_tm1 = pd_target = None
    manifolds = None

    if not use_random:
        theta_t = robot.default_theta()
        theta_tm1 = theta_t.clone()
        pd_target = theta_t.clone()
        print(f"\n--- warm-up ({warmup_steps} steps) ---")
        sim_output_save = sim._output
        if xl_only or energy_only:
            sim._output = False
        for s in range(warmup_steps):
            theta_tp1, _, manifolds = sim.step(
                theta_t, theta_tm1, pd_target, 100.0, 10.0)
            theta_tm1 = theta_t
            theta_t = theta_tp1
            print(f"  step {s + 1}: body_y={theta_t[1].item():.4f}  "
                  f"contacts={len(manifolds)}")
        sim._output = sim_output_save
        theta = theta_t.clone()

    if run_energy:
        print("\n" + "=" * 60)
        print("  debug_energy  (DDE / DDE-L / DDE-LL / DDE-P / DDE-XL)")
        print("=" * 60)
        sim.debug_energy(
            scale=scale, theta=theta, theta_t=theta_t, theta_tm1=theta_tm1,
            pd_target=pd_target, kp=100.0, kd=10.0, manifolds=manifolds)

    if run_backward:
        print("\n" + "=" * 60)
        print("  debug_backward (IFT)")
        print("=" * 60)
        sim.debug_backward(
            scale=scale, theta_t=theta_t, theta_tm1=theta_tm1,
            pd_target=pd_target, kp=100.0, kd=10.0, manifolds=manifolds)

    if run_theta_fd and theta is not None:
        print("\n" + "=" * 60)
        print("  debug_verify_theta_derivatives_fd")
        print("=" * 60)
        sim.debug_verify_theta_derivatives_fd(
            theta, theta_t, theta_tm1, pd_target,
            kp=100.0, kd=10.0, manifolds=manifolds)

    if run_xl_compare:
        if theta_t is None:
            print("\n[XL] use_random was set; XL test needs a warm-up state — skipping.")
        else:
            print("\n" + "=" * 60)
            print("  XL gradient (DTDXL) — ANA  vs  AD  vs  FD  (simulator_adapt)")
            print("=" * 60)
            print(
                f"  gtol={xl_gtol}  delta={xl_delta}  trials={xl_trials}  "
                f"autograd={'on' if not xl_no_autograd else 'off'}")
            run_dtdxl_compare(
                sim,
                theta_t=theta_t, theta_tm1=theta_tm1, pd_target=pd_target,
                manifolds=manifolds, kp=100.0, kd=10.0,
                delta=xl_delta, trials=xl_trials, head=xl_head,
                do_autograd=not xl_no_autograd, seed=xl_seed,
                banner=False)

    if render and theta is not None:
        from visualizer import SpiderVisualizer
        SpiderVisualizer(robot).snapshot(theta, title='Debug state')

    if energy_only:
        print("\nEnergy debug complete.")
    elif xl_only:
        print("\nXL debug complete.")
    else:
        print("\nFull debug complete.")


# =====================================================================
#  2. batch_debug — delegates per-env to single-env debug
# =====================================================================

def batch_debug(xml_path, device='cpu', N=4, *, use_random=False,
                scale=0.1, warmup_steps=10, n_envs=None):
    if not HAS_BATCH:
        print("ERROR: batch_simulator not available"); return

    print("=" * 60)
    print(f"  ConvexHullPBAD Batch Debug (N={N})")
    print("=" * 60)

    robot = Robot(xml_path, device=device, dtype=DTYPE)
    batch_sim = BatchSimulator(robot, N=N, dt=0.01, device=device, _output=False)

    ref = ConvexHullPBADSimulator(
        robot, dt=0.01, device=device, _output=False,
        barrier_x0=batch_sim.x0, coef_barrier=batch_sim.coef_barrier,
        lm_gamma=batch_sim.lm_gamma, friction=batch_sim.friction,
        gravity=batch_sim.gravity, barrier_d0=batch_sim.barrier_d0,
        implicit=batch_sim._implicit, use_friction=batch_sim._use_friction)

    ne = min(n_envs or N, N)

    if not use_random:
        theta_tb = _batch_initial_theta_spread_root_y(robot, N, device, DTYPE)
        theta_tmb = theta_tb.clone()
        pd_b = theta_tb.clone()
        print(f"\n--- batch warm-up ({warmup_steps} steps) ---")
        for s in range(warmup_steps):
            pd_one = pd_b.unsqueeze(0).expand(1, N, -1).contiguous()
            _, th_final = batch_sim.multi_step_batch(
                theta_tb, theta_tmb, pd_one, 100.0, 10.0)
            theta_tmb = theta_tb.clone()
            theta_tb = th_final.clone()
            print(f"  step {s + 1}: body_y_mean={theta_tb[:, 1].mean().item():.4f}")
        theta_b = theta_tb.clone()

        for e in range(ne):
            print(f"\n{'=' * 50} env {e}/{ne - 1} {'=' * 50}")
            th, tht, thtm = theta_b[e], theta_tb[e], theta_tmb[e]
            pd_ = pd_b[e]
            ref.debug_energy(scale=scale, theta=th, theta_t=tht,
                             theta_tm1=thtm, pd_target=pd_, kp=100.0, kd=10.0)
            ref.debug_backward(scale=scale, theta_t=tht, theta_tm1=thtm,
                               pd_target=pd_, kp=100.0, kd=10.0)
            ref.debug_verify_theta_derivatives_fd(
                th, tht, thtm, pd_, kp=100.0, kd=10.0)

    print("\nBatch debug complete.")


# =====================================================================
#  3. run_sim — single-env forward simulation (+ optional render)
# =====================================================================

def run_sim(xml_path, device='cpu', n_steps=200, dt=0.01, *, render=False):
    robot = Robot(xml_path, device=device, dtype=DTYPE)
    sim = ConvexHullPBADSimulator(robot, dt=dt, device=device)
    pd_seq = _default_pd_seq(robot, n_steps, device, DTYPE)

    theta_t = robot.default_theta()
    theta_tm1 = theta_t.clone()

    viz = plotter = link_data = None
    if render:
        from visualizer import SpiderVisualizer
        viz = SpiderVisualizer(robot)
        plotter, link_data = viz.open_plotter(theta_t)

    for step in range(n_steps):
        if viz:
            viz.wait_if_paused(plotter)
        pd = pd_seq[step].detach()
        t0 = time.time()
        theta_tp1, _, mfs = sim.step(theta_t, theta_tm1, pd, 100.0, 10.0)
        ms = (time.time() - t0) * 1000
        theta_tm1 = theta_t; theta_t = theta_tp1

        if viz:
            bp = theta_t[:3].detach().cpu().numpy()
            txt = (f"Step {step + 1}/{n_steps}\n"
                   f"Body: ({bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.3f})\n"
                   f"Contacts: {len(mfs)}  {ms:.0f}ms")
            if not viz.update_frame(plotter, link_data, theta_t, txt):
                break
            if time.time() - t0 < 1 / 30:
                time.sleep(1 / 30 - (time.time() - t0))
        elif (step + 1) % 10 == 0:
            print(f"  Step {step + 1}: y={theta_t[1].item():.4f}  "
                  f"K={len(mfs)}  {ms:.0f}ms")

    if plotter:
        plotter.close()
    print("Simulation complete.")


# =====================================================================
#  4. batch_run_sim — batch forward simulation (+ optional render)
# =====================================================================

def batch_run_sim(xml_path, device='cpu', N=4, n_steps=200, dt=0.01, *,
                  render=False):
    """Batch forward sim using one :meth:`BatchSimulator.multi_step_batch` call
    over the full ``n_steps``-row ``pd_seq``.

    Each env advances its own PD substep asynchronously inside LM.  With
    ``render=True``, the viewer refreshes whenever **any** env finishes a PD
    substep (``on_env_advance``).  Rendering uses a **stable** pose tensor: only
    rows for envs that just completed a substep are updated; others stay at
    their last shown pose so mid-LM trial/reject oscillation does not look like
    motion reversal.  With ``BatchSimulator._output`` True, LM logs use
    ``[LM env=k]`` on every environment.
    """
    if not HAS_BATCH:
        print("ERROR: batch_simulator not available"); return

    robot = Robot(xml_path, device=device, dtype=DTYPE)
    bsim = BatchSimulator(robot, N=N, dt=dt, device=device)

    theta_t = _batch_initial_theta_spread_root_y(robot, N, device, DTYPE)
    theta_tm1 = theta_t.clone()
    pd_seq = _default_pd_seq_batch(robot, N, n_steps, device, DTYPE, dt)
    pd_seq[:, :, 1] = theta_t[:, 1].unsqueeze(0).expand(n_steps, -1)

    viz = plotter = all_ld = x_off = z_off = None
    if render:
        from visualizer import SpiderVisualizer
        viz = SpiderVisualizer(robot)
        plotter, all_ld, x_off, z_off = viz.open_plotter_batch(
            N, theta_t)

    t0 = time.time()
    stop_viz = [False]

    if viz is not None:
        # One row per env: last *substep-complete* pose shown.  ``th_batch`` from
        # the sim still moves for envs mid-LM (rejected trials); do not draw that.
        theta_vis = [theta_t.clone()]

        def _on_env_advance(th_batch, step_idx_batch, advance_mask):
            if stop_viz[0]:
                return
            viz.wait_if_paused(plotter)
            adv = advance_mask.to(dtype=torch.bool)
            vis = theta_vis[0]
            vis_new = vis.clone()
            vis_new[adv] = th_batch[adv]
            theta_vis[0] = vis_new
            adv_ids = adv.nonzero(as_tuple=True)[0].tolist()
            idx_list = step_idx_batch.detach().cpu().tolist()
            txt = (
                f"PD substep advance  envs={adv_ids}  "
                f"completed_substeps={idx_list}  N={N}")
            if not viz.update_frame_batch(
                    plotter, all_ld, vis_new, x_off, z_off, txt):
                stop_viz[0] = True
                return
            time.sleep(0.02)

        _traj, th_final = bsim.multi_step_batch(
            theta_t, theta_tm1, pd_seq, 100.0, 10.0,
            on_env_advance=_on_env_advance)
    else:
        _traj, th_final = bsim.multi_step_batch(
            theta_t, theta_tm1, pd_seq, 100.0, 10.0)
    ms = (time.time() - t0) * 1000
    if viz is None:
        print(f"  multi_step_batch  n_steps={n_steps}  N={N}  wall={ms:.0f}ms")

    if plotter:
        plotter.close()
    print("Batch simulation complete.")


# =====================================================================
#  CLI
# =====================================================================

def _arg_kv(argv, key, cast=str, default=None):
    """Parse ``--key VALUE`` or ``--key=VALUE`` from ``argv``.

    Returns ``cast(value)`` if found, else ``default``.  Hand-rolled to keep the
    long-standing positional CLI (``main.py [mode] …``) compatible.
    """
    pref_eq = key + "="
    for i, a in enumerate(argv):
        if a == key and i + 1 < len(argv):
            try:
                return cast(argv[i + 1])
            except (TypeError, ValueError):
                return default
        if a.startswith(pref_eq):
            try:
                return cast(a[len(pref_eq):])
            except (TypeError, ValueError):
                return default
    return default


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    _here = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(_here, 'ant.xml')
    if not os.path.exists(xml_path):
        xml_path = os.path.join(_here, 'spider.xml')
    if not os.path.exists(xml_path):
        print(f"ERROR: no ant.xml or spider.xml under {_here}"); sys.exit(1)

    mode = sys.argv[1] if len(sys.argv) > 1 else 'sim'
    render = '--render' in sys.argv
    nums = [int(a) for a in sys.argv[2:] if a.isdigit()]

    # XL / energy options shared across `debug`, `debug_energy`, `debug_xl`.
    _dbg_kw = dict(
        energy_only='--energy-only' in sys.argv,
        xl='--xl' in sys.argv,
        xl_only='--xl-only' in sys.argv,
        xl_gtol=_arg_kv(sys.argv, '--gtol', float, 1e-12),
        xl_max_iter=_arg_kv(sys.argv, '--max-iter', int, 2000),
        xl_delta=_arg_kv(sys.argv, '--delta', float, 1e-5),
        xl_trials=_arg_kv(sys.argv, '--trials', int, 3),
        xl_head=_arg_kv(sys.argv, '--head', int, 8),
        xl_seed=_arg_kv(sys.argv, '--seed', int, 0),
        xl_no_autograd='--no-autograd' in sys.argv,
    )

    if mode == 'debug':
        debug(xml_path, device,
              use_random='random' in sys.argv, render=render,
              **_dbg_kw)
    elif mode == 'debug_energy':
        # Convenience alias: only DDE-* directional Hessian checks.
        kw = dict(_dbg_kw)
        kw['energy_only'] = True
        debug(xml_path, device,
              use_random='random' in sys.argv, render=render, **kw)
    elif mode == 'debug_xl':
        # Convenience alias: only DTDXL ANA / AD / FD compare.
        kw = dict(_dbg_kw)
        kw['xl_only'] = True
        debug(xml_path, device,
              use_random='random' in sys.argv, render=render, **kw)
    elif mode == 'batch_debug':
        batch_debug(xml_path, device, N=nums[0] if nums else 4,
                    use_random='random' in sys.argv)
    elif mode == 'sim':
        run_sim(xml_path, device, n_steps=nums[0] if nums else 200,
                render=render)
    elif mode == 'batch_sim':
        batch_run_sim(
            xml_path, device, N=nums[0] if nums else 4,
            n_steps=nums[1] if len(nums) > 1 else 200,
            render=render)
    else:
        print("Usage: python main.py [mode] [options]")
        print("  debug [random] [--render] [--energy-only] [--xl|--xl-only] [XL options]")
        print("        # default: simulator.* — debug_energy + debug_backward + θ-FD")
        print("        # --energy-only       : only debug_energy (every DDE-* line)")
        print("        # --xl                : simulator_adapt + extra DTDXL compare")
        print("        # --xl-only           : simulator_adapt + only DTDXL compare")
        print("  debug_energy [random] [--xl] [XL options]   (alias: debug --energy-only)")
        print("        # quick way to verify all DDE-* directional Hessians")
        print("  debug_xl [random] [XL options]              (alias: debug --xl-only)")
        print("  batch_debug [N] [random]")
        print("  sim [steps] [--render]")
        print("  batch_sim [N] [steps] [--render]")
        print("")
        print("XL options (default values shown):")
        print("  --gtol 1e-12        # LM gradient tolerance (tight ⇒ FD-IFT meaningful)")
        print("  --max-iter 2000     # LM iteration cap")
        print("  --delta 1e-5        # finite-difference step in vertex space")
        print("  --trials 3          # number of random unit dc directions")
        print("  --head 8            # leading components printed per vector")
        print("  --seed 0            # torch.manual_seed for dc directions")
        print("  --no-autograd       # disable AD reference (only ANA vs FD)")
