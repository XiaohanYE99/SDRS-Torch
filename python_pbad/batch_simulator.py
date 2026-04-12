"""
batch_simulator.py — Batched N-env parallel simulator

World frame is **Y-up** (same as robot XML / ``simulator.py``); gravity uses
vertex coordinate index 1.

Mirrors simulator.py's LM/Schur logic exactly, but vectorised across N
independent environments.  The batch dimension N is purely for GPU
parallelism — there is zero algorithmic coupling between environments.

Key design:
  - Each env has its own alpha, nu, E, g, H, p_warm, u_warm
  - Energy / gradient / Hessian computed analytically (no autograd)
  - Schur complement eliminates p, u → reduced [n x n] per env
  - LM trust-region accept/reject per env (identical to single-env);
    trial steps with contact penetration (d≤0 under fixed p) are rejected
  - ``multi_step_batch``: each **outer LM iteration** runs one batched Newton
    trial over all ``N`` envs.  ``pd_seq`` is ``[T, n]`` or ``[T, N, n]``.  Per-env
    PD substep index advances **asynchronously**.  With ``_output`` True, prints
    ``【s0,s1,...】`` each outer iter and per-env Levenberg–Marquardt lines tagged
    ``[LM env=k]`` (same style as :meth:`simulator.ConvexHullPBADSimulator._solve_lm`).
  - Recompute E, g, H when any env accepts a trial or advances a timestep
  - On each async PD substep advance, ``_p_warm`` / ``u_warm`` are cold-started
    from geometry (same role as single-env ``_detect_contacts`` at substep entry).
  - Optional ``on_env_advance`` callback on each PD substep advance (e.g. live
    rendering whenever any env finishes an LM substep).
"""

import math
import time
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Any, Dict, Callable
from robot import Robot
from simulator import barrier_eval, ConvexHullPBADSimulator


def _normalize_contact_barrier_mode(mode: str) -> str:
    m = str(mode).lower().strip()
    if m in ('acm', 'nested_acm'):
        return 'acm'
    return 'barrier'


def acm_nested_eval(d, x0, d0_offset, eta_l, r_l):
    """Multi-layer ACM-style barrier; vectorised over all layers at once."""
    L_acm = eta_l.shape[0]
    x0_base = float(x0)
    x0_layers = eta_l * x0_base                                     # [L_acm]

    # Expand d to [L_acm, *d.shape] for parallel barrier_eval across layers
    d_exp = d.unsqueeze(0).expand(L_acm, *d.shape)                  # [L_acm, ...]
    x_exp = d_exp - d0_offset
    clamp_min = 1e-15 if d.dtype == torch.float64 else 1e-12
    x_s = torch.clamp(x_exp, min=clamp_min)

    # x0_layers broadcast: [L_acm, 1, 1, ...] to match d dims
    shape_bc = [L_acm] + [1] * d.ndim
    x0_bc = x0_layers.reshape(shape_bc)

    u = x0_bc / x_s
    logv = torch.log(x_s / x0_bc)
    val_raw = -logv * (x_s - x0_bc) ** 2 / x_s
    grad_raw = -(1.0 - u) ** 2 - logv * (1.0 - u ** 2)
    hess_raw = (-(x_s - x0_bc) * (x_s + 3.0 * x0_bc)
                - 2.0 * x0_bc ** 2 * logv) / (x_s ** 3)

    active = (x_exp > 0) & (x_exp < x0_bc)
    zero = torch.zeros_like(x_exp)
    val_raw = torch.where(active, val_raw, zero)
    grad_raw = torch.where(active, grad_raw, zero)
    hess_raw = torch.where(active, hess_raw, zero)
    val_raw = torch.where(x_exp <= 0, torch.full_like(x_exp, 1e8), val_raw)

    r_bc = r_l.reshape(shape_bc)
    val = (r_bc * val_raw).sum(0)
    grad = (r_bc * grad_raw).sum(0)
    hess = (r_bc * hess_raw).sum(0)
    return val, grad, hess


# ============================================================================
#  Batch Simulator
# ============================================================================

class BatchSimulator:
    def __init__(
            self, robot: Robot, N: int = 1024,
            dt: float = 0.01,
            barrier_x0: float = 0.01,
            coef_barrier: float = 1e-2,
            lm_gamma: float = 1e0,
            friction: float = 0.8,
            gravity: float = -9.81,
            max_newton_iter: int = 1000,
            gtol: float = 1e-4,
            device='cuda',
            _output: bool = True,
            solver: str = 'lm',
            reject_step_energy_above: Optional[float] = 1e5,
            use_joint_limit_barrier: bool = False,
            barrier_d0: float = 1e-3,
            implicit: bool = True,
            use_friction: bool = True,
            contact_exclude_tree_distance: int = 4,
            friction_eps_s: Optional[float] = None,
            friction_eps_n: Optional[float] = None,
            contact_barrier: str = 'barrier',
            acm_r1: float = 1.0,
            acm_eta1: Optional[float] = None,
            acm_n_layers: int = 32,
            contact_distance_mode: str = 'barrier'):
        """
        Same physics / LM knobs as :class:`simulator.ConvexHullPBADSimulator`,
        plus batch-only ``N``.

        ``_output`` toggles diagnostics: pair counts on init; in :meth:`multi_step_batch`,
        each-outer-iter ``【step_idx per env…】``, per-env ``[LM env=k]`` / ``iter`` /
        ``converged`` lines; and ``[multi] done``.

        ``contact_distance_mode`` is kept for API compatibility: contact activation
        always matches single-env :meth:`~simulator.ConvexHullPBADSimulator._detect_contacts`
        (barrier energy sum ``> 0``).  Legacy values ``'aabb'`` / ``'plane'`` are
        accepted and ignored.
        """
        self.robot = robot
        self.N = N
        self.dt = dt
        self.x0 = barrier_x0
        self.barrier_d0 = float(barrier_d0)
        self._barrier_d0_half = self.barrier_d0 * 0.5
        self._contact_barrier_mode = _normalize_contact_barrier_mode(contact_barrier)
        self.acm_r1 = float(acm_r1)
        self.acm_eta1 = None if acm_eta1 is None else float(acm_eta1)
        self.acm_n_layers = int(acm_n_layers)
        self.coef_barrier = coef_barrier
        self.lm_gamma = lm_gamma
        self.friction = friction
        self.gravity = gravity
        self.max_iter = max_newton_iter
        self.gtol = gtol
        self.device = device
        self.dtype = torch.float64
        self.n_dof = robot.n_dof
        self._output = _output
        self._reject_step_energy_above = (
            None if reject_step_energy_above is None
            else float(reject_step_energy_above))
        self._use_joint_limit_barrier = bool(use_joint_limit_barrier)
        self._contact_exclude_tree_distance = max(1, int(contact_exclude_tree_distance))
        sol = str(solver).lower().strip()
        if sol not in ('lm', 'newton_ls'):
            raise ValueError("solver must be 'lm' or 'newton_ls'")
        self._solver = sol
        cdm = str(contact_distance_mode).lower().strip()
        if cdm in ('vertex_gpu', 'gjk'):
            cdm = 'aabb'
        if cdm not in ('barrier', 'single', 'energy', 'aabb', 'plane'):
            raise ValueError(
                "contact_distance_mode must be 'barrier' (default, single-env "
                "parity), or legacy 'aabb' / 'plane' (ignored).")
        if cdm in ('aabb', 'plane') and _output:
            print(
                "  [BatchSim] contact_distance_mode='"
                f"{contact_distance_mode}' is ignored; "
                "using barrier activation (same as ConvexHullPBADSimulator).")
        self._contact_distance_mode = 'barrier'
        self._implicit = bool(implicit)
        self._use_friction = bool(use_friction)
        self._friction_eps_s = (
            float(friction_eps_s) if friction_eps_s is not None else 1e-4)
        self._friction_eps_n = (
            float(friction_eps_n) if friction_eps_n is not None else 1e-8)

        for link in self.robot.links:
            link.local_vertices = link.local_vertices.to(dtype=self.dtype, device=device)
        for j in self.robot.joints:
            j.origin = j.origin.to(dtype=self.dtype, device=device)
            j.axis = j.axis.to(dtype=self.dtype, device=device)
        if self.robot.ground is not None:
            self.robot.ground.vertices = self.robot.ground.vertices.to(
                dtype=self.dtype, device=device)
        if self.robot.initial_pos is not None:
            self.robot.initial_pos = self.robot.initial_pos.to(
                dtype=self.dtype, device=device)
        self.robot.dtype = self.dtype

        self._precompute_cache()
        self._precompute_contact_pairs()

    # ==================================================================
    #  Cache
    # ==================================================================
    def _precompute_cache(self):
        dev, dty = self.device, self.dtype
        N = self.N
        L = len(self.robot.links)
        n = self.n_dof
        self._L = L
        self._n = n

        vert_counts = [lk.local_vertices.shape[0] for lk in self.robot.links]
        max_M = max(vert_counts)
        self._max_M = max_M
        self._vert_counts = torch.tensor(vert_counts, device=dev, dtype=torch.long)

        padded = []
        for lk in self.robot.links:
            v = lk.local_vertices
            if v.shape[0] < max_M:
                pad = torch.zeros(max_M - v.shape[0], 3, device=dev, dtype=dty)
                v = torch.cat([v, pad], dim=0)
            padded.append(v)
        self._local_verts = torch.stack(padded)                # [L, mM, 3]

        mask = torch.zeros(L, max_M, device=dev, dtype=torch.bool)
        for i, c in enumerate(vert_counts):
            mask[i, :c] = True
        self._vert_mask = mask
        self._vmask_f = mask.to(dty)

        self._rho = torch.tensor(
            [lk.mass / lk.local_vertices.shape[0]
             for lk in self.robot.links], device=dev, dtype=dty)
        self._rho_2d = self._rho.view(L, 1)
        self._rho_3d = self._rho.view(L, 1, 1)

        gv = self.robot.ground.vertices
        self._ground_h = torch.cat(
            [gv, torch.ones(gv.shape[0], 1, device=dev, dtype=dty)], 1)

        self._eye3 = torch.eye(3, device=dev, dtype=dty)
        self._eye4 = torch.eye(4, device=dev, dtype=dty)
        self._eye_n = torch.eye(n, device=dev, dtype=dty)

        self.joint_mask = torch.zeros(n, device=dev, dtype=dty)
        if n > 6:
            self.joint_mask[6:] = 1.0

        h_off, h_lo, h_hi = [], [], []
        for j in self.robot.joints:
            if j.jtype == 'hinge':
                h_off.append(j.dof_offset)
                h_lo.append(j.limit_lower)
                h_hi.append(j.limit_upper)
        if h_off:
            self._hinge_idx = torch.tensor(h_off, device=dev, dtype=torch.long)
            self._hinge_lo = torch.tensor(h_lo, device=dev, dtype=dty)
            self._hinge_hi = torch.tensor(h_hi, device=dev, dtype=dty)
        else:
            self._hinge_idx = None

        self._descendants = {}
        self._desc_tensors = {}
        for ji, joint in enumerate(self.robot.joints):
            desc = set()
            queue = [joint.child_link]
            while queue:
                lk = queue.pop()
                desc.add(lk)
                for jj, j2 in enumerate(self.robot.joints):
                    if j2.parent_link == lk:
                        queue.append(j2.child_link)
            self._descendants[ji] = sorted(desc)
            self._desc_tensors[ji] = torch.tensor(
                sorted(desc), device=dev, dtype=torch.long)

        self._ancestors = {}
        for ji in range(len(self.robot.joints)):
            anc = []
            cur_link = self.robot.joints[ji].parent_link
            while cur_link >= 0:
                for jj, j2 in enumerate(self.robot.joints):
                    if j2.child_link == cur_link:
                        anc.append(jj)
                        cur_link = j2.parent_link
                        break
                else:
                    break
            anc.reverse()
            self._ancestors[ji] = anc

        L_acm = max(1, int(self.acm_n_layers))
        idx_a = torch.arange(1, L_acm + 1, device=dev, dtype=dty)
        _e1 = float(self.acm_eta1) if self.acm_eta1 is not None else float(self.x0)
        self._acm_eta1_eff = _e1
        self._acm_eta_l = _e1 * (idx_a ** (-0.25))
        self._acm_r_l = float(self.acm_r1) * (idx_a ** 3)

    def _contact_barrier_eval(self, d, x0, d0_offset):
        if self._contact_barrier_mode == 'acm':
            return acm_nested_eval(
                d, x0, d0_offset, self._acm_eta_l, self._acm_r_l)
        return barrier_eval(d, x0, d0_offset)

    # ------------------------------------------------------------------
    #  Friction helpers  (batched mirrors of simulator.py)
    # ------------------------------------------------------------------
    def _friction_tangent_basis_batch(self, n_hat):
        """Orthonormal tangent basis ``t0, t1 ⟂ n_hat``.  [..., 3] → ([], [])."""
        dev, dty = n_hat.device, n_hat.dtype
        shape = n_hat.shape[:-1]
        ref = torch.zeros(*shape, 3, device=dev, dtype=dty)
        ref[..., 2] = 1.0
        t0 = torch.cross(n_hat, ref, dim=-1)
        bad = t0.norm(dim=-1) < 1e-10
        if bad.any():
            ref2 = torch.zeros_like(ref)
            ref2[..., 0] = 1.0
            t0[bad] = torch.cross(n_hat[bad], ref2[bad], dim=-1)
        t0 = t0 / (t0.norm(dim=-1, keepdim=True) + 1e-30)
        t1 = torch.cross(n_hat, t0, dim=-1)
        return t0, t1

    def _unified_u_to_xyz_omega_batch(self, u, n_hat):
        """``u`` [...,3] → ``u_xyz`` [...,3], ``omega`` [...], ``t0``, ``t1``."""
        t0, t1 = self._friction_tangent_basis_batch(n_hat)
        u_xyz = u[..., 0:1] * t0 + u[..., 1:2] * t1
        omega = u[..., 2]
        return u_xyz, omega, t0, t1

    @staticmethod
    def _friction_J_lift_batch(t0, t1):
        """``(t0,t1)`` [...,3] → ``J`` [...,4,3]."""
        J = torch.zeros(*t0.shape[:-1], 4, 3, device=t0.device, dtype=t0.dtype)
        J[..., :3, 0] = t0
        J[..., :3, 1] = t1
        J[..., 3, 2] = 1.0
        return J

    @staticmethod
    def _friction_reduce_g_batch(g_xyz, g_om, J):
        """``g4=[g_xyz|g_om]`` → ``J^T g4``."""
        g4 = torch.cat([g_xyz, g_om.unsqueeze(-1)], dim=-1)
        return torch.einsum('...ji,...j->...i', J, g4)

    @staticmethod
    def _friction_reduce_H_batch(H4, J):
        """``J^T H4 J``."""
        return torch.matmul(J.transpose(-1, -2), torch.matmul(H4, J))

    @staticmethod
    def _friction_reduce_Htu_batch(Htu4, J):
        """``Htu4`` [...,n,4] → ``Htu`` [...,n,3]."""
        return torch.einsum('...nj,...ji->...ni', Htu4, J)

    # ==================================================================
    #  Contact pair precomputation — same lists + plane init as single-env
    #  :meth:`simulator.ConvexHullPBADSimulator._precompute_contact_pairs`
    #  (batched GPU pass via ``_init_separating_planes_batch``).
    # ==================================================================
    def _va_vb_lists_from_world_verts(
            self, wv0: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """One configuration: world verts ``[L, mM, 3]`` → ``(va_list, vb_list)`` for
        :meth:`ConvexHullPBADSimulator._init_separating_planes_batch`."""
        L = self._L
        vc = self._vert_counts
        excluded = self._excluded_link_pairs

        va_list: List[torch.Tensor] = []
        vb_list: List[torch.Tensor] = []

        for i in range(L):
            for j in range(i + 1, L):
                if (i, j) in excluded:
                    continue
                ni = int(vc[i].item())
                nj = int(vc[j].item())
                va_list.append(wv0[i, :ni].clone())
                vb_list.append(wv0[j, :nj].clone())

        gv = self.robot.ground.vertices if self.robot.ground is not None else None
        if gv is not None:
            for i in range(L):
                ni = int(vc[i].item())
                va_list.append(wv0[i, :ni].clone())
                vb_list.append(gv.clone())

        return va_list, vb_list

    @torch.no_grad()
    def _reinit_p_warm_from_theta_batch(self, theta: torch.Tensor) -> None:
        """Per-env warm planes from FK(θ); resets friction multipliers.

        ``_precompute_contact_pairs`` builds planes from ``default_theta`` and
        ``expand``\ s them to all envs.  Different per-env poses (e.g. spread
        root Y) need their own initial ``p`` or barrier distances become invalid
        and energy spikes (``x<=0`` → large penalties).
        """
        if self._P == 0:
            return
        dev, dty = self.device, self.dtype
        if theta.shape[0] != self.N:
            raise ValueError(
                f"_reinit_p_warm_from_theta_batch: theta N={theta.shape[0]} "
                f"!= BatchSimulator.N={self.N}")
        wv = self._get_wv_batch(theta)
        helper = ConvexHullPBADSimulator.__new__(ConvexHullPBADSimulator)
        helper.device = dev
        helper.dtype = dty
        for e in range(self.N):
            va_list, vb_list = self._va_vb_lists_from_world_verts(wv[e])
            p_e = ConvexHullPBADSimulator._init_separating_planes_batch(
                helper, va_list, vb_list)
            self._p_warm[e].copy_(p_e)
        self._u_warm.zero_()

    def _precompute_contact_pairs(self):
        dev, dty = self.device, self.dtype
        N, L = self.N, self._L

        from simulator import ConvexHullPBADSimulator as _CHS
        self._excluded_link_pairs = _CHS._excluded_link_pairs_floyd(
            self.robot, self._contact_exclude_tree_distance)

        theta0 = self.robot.default_theta().to(dtype=dty, device=dev)
        with torch.no_grad():
            wv0 = self._get_wv_single(theta0)

        va_list, vb_list = self._va_vb_lists_from_world_verts(wv0)
        pair_tags: List[tuple] = []

        L = self._L
        for i in range(L):
            for j in range(i + 1, L):
                if (i, j) in self._excluded_link_pairs:
                    continue
                pair_tags.append(('link', i, j))

        gv = self.robot.ground.vertices if self.robot.ground is not None else None
        if gv is not None:
            for i in range(L):
                pair_tags.append(('ground', i))

        if len(va_list) == 0:
            P = 0
            self._P = P
            self._lid_a = torch.zeros(0, dtype=torch.long, device=dev)
            self._lid_b = torch.zeros(0, dtype=torch.long, device=dev)
            self._is_ground = torch.zeros(0, dtype=torch.bool, device=dev)
            self._gnd_idx = torch.zeros(0, dtype=torch.long, device=dev)
            self._lnk_idx = torch.zeros(0, dtype=torch.long, device=dev)
            self._pair_key_to_idx = {}
            self._scatter_a = torch.zeros(0, dtype=torch.long, device=dev)
            self._p_warm = torch.zeros(N, 0, 4, device=dev, dtype=dty)
            self._u_warm = torch.zeros(N, 0, 3, device=dev, dtype=dty)
            self._alpha = torch.full((N,), 1e-6, device=dev, dtype=dty)
            if self._output:
                print(f"  [BatchSim] N={N}  pairs=0  (no contact pairs)")
            return

        helper = ConvexHullPBADSimulator.__new__(ConvexHullPBADSimulator)
        helper.device = dev
        helper.dtype = dty
        p_all = ConvexHullPBADSimulator._init_separating_planes_batch(
            helper, va_list, vb_list)

        lid_a_list: List[int] = []
        lid_b_list: List[int] = []
        for tag in pair_tags:
            if tag[0] == 'link':
                _, i, j = tag
                lid_a_list.append(i)
                lid_b_list.append(j)
            else:
                _, i = tag
                lid_a_list.append(i)
                lid_b_list.append(-1)

        lid_a = torch.tensor(lid_a_list, device=dev, dtype=torch.long)
        lid_b = torch.tensor(lid_b_list, device=dev, dtype=torch.long)
        p_init = p_all
        P = int(p_init.shape[0])
        self._P = P

        self._lid_a = lid_a
        self._lid_b = lid_b
        self._is_ground = (lid_b < 0)
        self._gnd_idx = self._is_ground.nonzero(as_tuple=True)[0]
        self._lnk_idx = (~self._is_ground).nonzero(as_tuple=True)[0]

        self._pair_key_to_idx = {}
        for k in range(P):
            la = int(lid_a[k].item())
            lb = int(lid_b[k].item())
            self._pair_key_to_idx[(la, lb)] = k

        self._scatter_a = lid_a
        if self._lnk_idx.numel() > 0:
            self._scatter_b_lnk = self._lid_b[self._lnk_idx]
            lnk_local = torch.full((P,), -1, device=dev, dtype=torch.long)
            lnk_local[self._lnk_idx] = torch.arange(
                self._lnk_idx.numel(), device=dev, dtype=torch.long)
            self._lnk_local = lnk_local

        self._p_warm = p_init.unsqueeze(0).expand(N, -1, -1).clone()
        self._u_warm = torch.zeros(N, P, 3, device=dev, dtype=dty)
        self._alpha = torch.full((N,), 1e-6, device=dev, dtype=dty)

        if self._output:
            n_ll = int(self._lnk_idx.numel())
            n_g = int(self._gnd_idx.numel())
            print(f"  [BatchSim] N={N}  pairs={P}  "
                  f"link-link={n_ll}  link-ground={n_g}  "
                  f"(planes via ConvexHullPBADSimulator._init_separating_planes_batch)")

    # ==================================================================
    #  World vertices  [N, L, mM, 3]
    # ==================================================================
    def _get_wv_batch(self, theta):
        T = self.robot.forward_kinematics_batch(theta)
        R = T[:, :, :3, :3]
        t = T[:, :, :3, 3]
        return torch.einsum('nlij,lmj->nlmi', R, self._local_verts) + t.unsqueeze(2)

    def _get_wv_single(self, theta):
        transforms = self.robot.forward_kinematics(theta)
        T = torch.stack(transforms)
        R = T[:, :3, :3]
        t = T[:, :3, 3]
        return torch.einsum('lmj,lkj->lmk', self._local_verts, R) + t.unsqueeze(1)

    # ==================================================================
    #  FK Jacobian (analytic, batched) [N, L, mM, 3, n]
    # ==================================================================
    def _compute_fk_jacobian_batch(self, theta):
        """Returns ``(wv, J, T_all, dR_stack_cache)``.

        ``T_all`` and ``dR_stack_cache`` are cached so that
        ``_compute_fk_correction_batch`` can reuse them without a second FK
        pass or redundant trig evaluation.
        """
        N, n = theta.shape
        dev, dty = self.device, self.dtype
        L, mM = self._L, self._max_M

        dR_cache = {}

        with torch.no_grad():
            T_all = self.robot.forward_kinematics_batch(theta.detach())
            R_all = T_all[:, :, :3, :3]
            t_all = T_all[:, :, :3, 3]
            wv = (torch.einsum('nlij,lmj->nlmi', R_all, self._local_verts)
                  + t_all.unsqueeze(2))
            J = torch.zeros(N, L, mM, 3, n, device=dev, dtype=dty)

            for ji, joint in enumerate(self.robot.joints):
                off = joint.dof_offset
                parent = joint.parent_link
                desc_links = self._descendants[ji]
                if not desc_links:
                    continue
                desc = self._desc_tensors[ji]
                wv_desc = wv[:, desc]
                D = len(desc_links)

                R_par = (self._eye3.unsqueeze(0).expand(N, -1, -1)
                         if parent < 0 else T_all[:, parent, :3, :3])
                t_par = (torch.zeros(N, 3, device=dev, dtype=dty)
                         if parent < 0 else T_all[:, parent, :3, 3])

                if joint.jtype == 'hinge':
                    a_norm = joint.axis / (joint.axis.norm() + 1e-12)
                    a_w = torch.einsum('nij,j->ni', R_par, a_norm)
                    o_w = torch.einsum('nij,j->ni', R_par, joint.origin) + t_par
                    diff = wv_desc - o_w.reshape(N, 1, 1, 3)
                    cross = torch.linalg.cross(
                        a_w.reshape(N, 1, 1, 3).expand_as(diff), diff)
                    J[:, desc, :, :, off] = cross

                elif joint.jtype == 'free':
                    J[:, desc, :, :, off:off + 3] = R_par.reshape(
                        N, 1, 1, 3, 3).expand(N, D, mM, -1, -1)

                    roll  = theta[:, off + 3].detach()
                    pitch = theta[:, off + 4].detach()
                    yaw   = theta[:, off + 5].detach()
                    cr, sr = torch.cos(roll),  torch.sin(roll)
                    cp, sp = torch.cos(pitch), torch.sin(pitch)
                    cy, sy = torch.cos(yaw),   torch.sin(yaw)
                    z = torch.zeros_like(cr)

                    dR_stack = torch.stack([
                        torch.stack([
                            torch.stack([z, cy*sp*cr+sy*sr, -cy*sp*sr+sy*cr], 1),
                            torch.stack([z, sy*sp*cr-cy*sr, -sy*sp*sr-cy*cr], 1),
                            torch.stack([z, cp*cr, -cp*sr], 1)], 1),
                        torch.stack([
                            torch.stack([-cy*sp, cy*cp*sr, cy*cp*cr], 1),
                            torch.stack([-sy*sp, sy*cp*sr, sy*cp*cr], 1),
                            torch.stack([-cp, -sp*sr, -sp*cr], 1)], 1),
                        torch.stack([
                            torch.stack([-sy*cp, -sy*sp*sr-cy*cr, -sy*sp*cr+cy*sr], 1),
                            torch.stack([cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], 1),
                            torch.stack([z, z, z], 1)], 1)], 0)  # [3,N,3,3]
                    dR_stack = dR_stack.permute(1, 0, 2, 3)           # [N,3,3,3]

                    dR_cache[ji] = (cr, sr, cp, sp, cy, sy, z, dR_stack)

                    child = joint.child_link
                    R_child = T_all[:, child, :3, :3]
                    t_child = T_all[:, child, :3, 3]
                    q = torch.einsum('nij,ndmj->ndmi', R_child.transpose(1,2),
                                     wv_desc - t_child.reshape(N, 1, 1, 3))
                    RdR = torch.einsum('nij,nejk->neik', R_par, dR_stack)
                    J_cols = torch.einsum('neij,ndmj->nedmi', RdR, q)
                    J[:, desc, :, :, off + 3:off + 6] = J_cols.permute(0, 2, 3, 4, 1)

        return wv, J, T_all, dR_cache

    # ==================================================================
    #  Contact detection  [N, P] bool
    #  Same rule as :meth:`simulator.ConvexHullPBADSimulator._detect_contacts`:
    #  per pair, ``val_a.sum() + val_b.sum() > 0`` (barrier energies).
    #
    #  ``p_planes`` must match the same warm planes used for detection in
    #  single-env: normally ``_p_warm``.  On LM **trial** with Schur
    #  (implicit=True), single-env updates ``_pair_manifolds`` to back-subbed
    #  ``p`` *before* :meth:`_detect_contacts` — pass ``p_try`` here so batch
    #  matches (see ``_solve_lm`` after ``_back_substitute``).
    # ==================================================================
    def _detect_contacts_batch(self, theta,
                               p_planes: Optional[torch.Tensor] = None):
        dev, dty = self.device, self.dtype
        N, P = self.N, self._P
        mM = self._max_M
        if P == 0:
            return torch.zeros(N, 0, device=dev, dtype=torch.bool)

        x0 = self.x0
        d0h = self._barrier_d0_half
        vmask = self._vmask_f

        with torch.no_grad():
            wv = self._get_wv_batch(theta)
            pk_all = self._p_warm if p_planes is None else p_planes

        # Side-a: all pairs use padded vertices, mask out padding
        va = wv[:, self._lid_a]                                     # [N,P,mM,3]
        ones_a = torch.ones(N, P, mM, 1, device=dev, dtype=dty)
        va_h = torch.cat([va, ones_a], dim=3)                       # [N,P,mM,4]
        d_a = -torch.einsum('npmi,npi->npm', va_h, pk_all)          # [N,P,mM]
        val_a, _, _ = self._contact_barrier_eval(d_a, x0, d0h)
        vmask_a = vmask[self._lid_a]                                 # [P,mM]
        sum_a = (val_a * vmask_a.unsqueeze(0)).sum(dim=2)            # [N,P]

        # Side-b: ground pairs and link-link pairs
        sum_b = torch.zeros(N, P, device=dev, dtype=dty)
        if self._gnd_idx.numel() > 0:
            gi = self._gnd_idx
            d_b_gnd = torch.einsum('gi,npi->npg',
                                   self._ground_h, pk_all[:, gi])
            val_b_gnd, _, _ = self._contact_barrier_eval(d_b_gnd, x0, d0h)
            sum_b[:, gi] = val_b_gnd.sum(dim=2)

        if self._lnk_idx.numel() > 0:
            li = self._lnk_idx
            lid_b_lnk = self._lid_b[li]
            vb = wv[:, lid_b_lnk]                                   # [N,Pl,mM,3]
            ones_l = torch.ones(N, li.numel(), mM, 1, device=dev, dtype=dty)
            vb_h = torch.cat([vb, ones_l], dim=3)
            d_b_lnk = torch.einsum('npmi,npi->npm', vb_h, pk_all[:, li])
            val_b_lnk, _, _ = self._contact_barrier_eval(d_b_lnk, x0, d0h)
            vmask_b = vmask[lid_b_lnk]
            sum_b[:, li] = (val_b_lnk * vmask_b.unsqueeze(0)).sum(dim=2)

        return (sum_a + sum_b) > 0.0

    @torch.no_grad()
    def _penetration_fixed_pu_batch(self, theta, p_vals, active_mask):
        """Per-env True if any **active** contact has d≤0 (same convention as simulator).

        Uses fixed planes ``p_vals`` [N,P,4] at ``theta``, matching
        :meth:`_eval_energy_batch` trial evaluation.
        """
        dev, dty = self.device, self.dtype
        N, P = self.N, self._P
        if P == 0 or not active_mask.any():
            return torch.zeros(N, device=dev, dtype=torch.bool)

        wv = self._get_wv_batch(theta)
        va = wv[:, self._lid_a]
        ones = torch.ones(N, P, self._max_M, 1, device=dev, dtype=dty)
        va_h = torch.cat([va, ones], 3)
        d_a = -torch.einsum('npmi,npi->npm', va_h, p_vals)
        vmask_a = self._vmask_f[self._lid_a]
        d_a_masked = d_a + (1 - vmask_a.unsqueeze(0)) * 1e6
        min_da = d_a_masked.amin(2)

        min_db = torch.full((N, P), 1e6, device=dev, dtype=dty)
        if self._gnd_idx.numel() > 0:
            gi = self._gnd_idx
            d_b_gnd = torch.einsum('gi,npi->npg', self._ground_h,
                                   p_vals[:, gi])
            min_db[:, gi] = d_b_gnd.amin(2)

        if self._lnk_idx.numel() > 0:
            li = self._lnk_idx
            lid_b_lnk = self._lid_b[li]
            vb = wv[:, lid_b_lnk]
            ones_l = torch.ones(N, li.numel(), self._max_M, 1,
                                device=dev, dtype=dty)
            vb_h = torch.cat([vb, ones_l], 3)
            d_b_lnk = torch.einsum('npmi,npi->npm', vb_h, p_vals[:, li])
            vmask_b = self._vmask_f[lid_b_lnk]
            d_b_lnk_masked = d_b_lnk + (1 - vmask_b.unsqueeze(0)) * 1e6
            min_db[:, li] = d_b_lnk_masked.amin(2)

        bad = (min_da <= 0) | (min_db <= 0)
        amask = active_mask.to(torch.bool)
        return (bad & amask).any(dim=1)

    # ==================================================================
    #  Eval energy only (for trial steps)  [N]
    # ==================================================================
    def _eval_energy_batch(self, theta, p_vals, u_vals, active_mask,
                           p_normals, wv_curr, wv_last, pd, theta_t,
                           kp, kd):
        dev, dty = self.device, self.dtype
        N, n = theta.shape
        P, mM = self._P, self._max_M
        coef, x0, dt = self.coef_barrier, self.x0, self.dt
        vmask = self._vmask_f
        amask_f = active_mask.to(dty)

        with torch.no_grad():
            wv_next = self._get_wv_batch(theta)

        accel = wv_next - 2.0 * wv_curr + wv_last
        E = (0.5 / (dt * dt)) * (self._rho_3d * vmask.unsqueeze(2) * accel.pow(2)).sum(dim=(1, 2, 3))
        E = E - (self._rho_2d * self.gravity * wv_next[:, :, :, 1] * vmask).sum(dim=(1, 2))

        mask = self.joint_mask
        pos_err = (theta - pd) * mask
        vel_err = (theta - theta_t) * mask
        E = E + 0.5 * kp * pos_err.pow(2).sum(1) + 0.5 * kd * vel_err.pow(2).sum(1)

        if self._use_joint_limit_barrier and self._hinge_idx is not None:
            q = theta[:, self._hinge_idx]
            vl, _, _ = barrier_eval(q - self._hinge_lo, 0.3)
            vu, _, _ = barrier_eval(self._hinge_hi - q, 0.3)
            E = E + (vl + vu).sum(1) * 10.0

        if P == 0 or not active_mask.any():
            return E

        va = wv_next[:, self._lid_a]
        ones = torch.ones(N, P, mM, 1, device=dev, dtype=dty)
        va_h = torch.cat([va, ones], 3)
        vmask_a = vmask[self._lid_a]

        d_a = -torch.einsum('npmi,npi->npm', va_h, p_vals)
        val_a, _, _ = self._contact_barrier_eval(d_a, x0, self._barrier_d0_half)
        val_a = val_a * vmask_a.unsqueeze(0) * amask_f.unsqueeze(2)

        norm_p = torch.norm(p_vals[:, :, :3], dim=2)
        val_n, _, _ = barrier_eval(1.0 - norm_p, x0)
        E = E + coef * (val_a.sum(dim=(1, 2)) + (val_n * amask_f).sum(1))

        if self._gnd_idx.numel() > 0:
            gi = self._gnd_idx
            amask_gnd = amask_f[:, gi]
            d_b_gnd = torch.einsum('gi,npi->npg', self._ground_h, p_vals[:, gi])
            val_b_gnd, _, _ = self._contact_barrier_eval(d_b_gnd, x0, self._barrier_d0_half)
            E = E + coef * (val_b_gnd * amask_gnd.unsqueeze(2)).sum(dim=(1, 2))

        if self._lnk_idx.numel() > 0:
            li = self._lnk_idx
            amask_ll = amask_f[:, li]
            lid_b_lnk = self._lid_b[li]
            vb = wv_next[:, lid_b_lnk]
            ones_l = torch.ones(N, li.numel(), mM, 1, device=dev, dtype=dty)
            vb_h = torch.cat([vb, ones_l], 3)
            vmask_b = vmask[lid_b_lnk]
            d_b_lnk = torch.einsum('npmi,npi->npm', vb_h, p_vals[:, li])
            val_b_lnk, _, _ = self._contact_barrier_eval(d_b_lnk, x0, self._barrier_d0_half)
            val_b_lnk = val_b_lnk * vmask_b.unsqueeze(0) * amask_ll.unsqueeze(2)
            E = E + coef * val_b_lnk.sum(dim=(1, 2))

        # Friction energy (matches simulator._friction_energy_total_pn)
        if self._use_friction:
            _eps_n = self._friction_eps_n
            _eps_s = self._friction_eps_s
            n_vecs = p_normals[:, :, :3]
            norm_n = torch.norm(n_vecs, dim=2, keepdim=True) + _eps_n
            n_hat = n_vecs / norm_n
            Proj = self._eye3 - torch.einsum('npi,npj->npij', n_hat, n_hat)

            u_xyz, omega_e, _, _ = self._unified_u_to_xyz_omega_batch(
                u_vals, n_hat)

            va_curr = wv_curr[:, self._lid_a]
            vel_a = (va - va_curr) / dt
            vel = vel_a.clone()
            if self._lnk_idx.numel() > 0:
                li = self._lnk_idx
                lid_b_lnk = self._lid_b[li]
                vb_curr = wv_curr[:, lid_b_lnk]
                vel[:, li] = vel[:, li] - (wv_next[:, lid_b_lnk] - vb_curr) / dt

            tan_vel = torch.einsum('npij,npmj->npmi', Proj, vel)
            r_nx_e = torch.cross(
                n_hat.unsqueeze(2).expand(-1, -1, mM, -1), va_curr, dim=3)
            omega_term_e = omega_e.unsqueeze(2).unsqueeze(3) * r_nx_e
            rel_vel = tan_vel - u_xyz.unsqueeze(2) - omega_term_e

            ones_c = torch.ones(N, P, mM, 1, device=dev, dtype=dty)
            va_h_c = torch.cat([va_curr, ones_c], 3)
            d_fn = -torch.einsum('npmi,npi->npm', va_h_c, p_normals)
            _, bg_fn, _ = self._contact_barrier_eval(d_fn, x0, self._barrier_d0_half)
            pn3_e = p_normals[:, :, :3]
            f_vec_e = coef * bg_fn.unsqueeze(3) * pn3_e.unsqueeze(2)
            fn_sq_e = f_vec_e.pow(2).sum(dim=3)
            sqrt_eps_e = torch.sqrt(torch.tensor(_eps_s, device=dev, dtype=dty))
            A_m_e = torch.sqrt(fn_sq_e + _eps_s) - sqrt_eps_e
            link_mask_e = vmask_a.unsqueeze(0) * amask_f.unsqueeze(2)

            s_norm = torch.sqrt(rel_vel.pow(2).sum(3) + _eps_s)
            E = E + (self.friction * dt * A_m_e * s_norm * link_mask_e
                     ).sum(dim=(1, 2))

        return E

    # ==================================================================
    #  Compute all: E, g_theta, H_theta + manifold Hessians
    #  Fully vectorised — no Python loops over P or N.
    #
    #  ``p_normals`` [N,P,4]: friction planes only — frozen at substep start
    #  (``θ_t`` / C++ ``manifoldsLast``), matching :meth:`simulator.ConvexHullPBADSimulator.step`
    #  + ``friction_plane_snap``.  Contact barriers use ``self._p_warm`` (current iterate).
    # ==================================================================
    def _compute_all_batch(self, theta, active_mask, p_normals,
                           wv_curr, wv_last, pd, theta_t, kp, kd):
        dev, dty = self.device, self.dtype
        N, n = theta.shape
        L, P, mM = self._L, self._P, self._max_M
        dt = self.dt
        coef, x0 = self.coef_barrier, self.x0
        vmask = self._vmask_f
        amask_f = active_mask.to(dty)
        _t = {}
        _t0 = time.perf_counter

        # ---- FK + Jacobian ----
        t_ = _t0()
        wv_next, J, T_all, dR_cache = self._compute_fk_jacobian_batch(theta)
        _t['fk'] = _t0() - t_

        # ---- Inertial E, g_v, H_vv ----
        t_ = _t0()
        accel = wv_next - 2.0 * wv_curr + wv_last
        E = (0.5 / (dt * dt)) * (self._rho_3d * vmask.unsqueeze(2) * accel.pow(2)).sum(dim=(1, 2, 3))
        E = E - (self._rho_2d * self.gravity * wv_next[:, :, :, 1] * vmask).sum(dim=(1, 2))
        rho_vm = (self._rho_2d * vmask).unsqueeze(0)
        g_v = ((1.0 / (dt * dt)) * rho_vm.unsqueeze(3) * accel)
        g_v[:, :, :, 1] = g_v[:, :, :, 1] - (self._rho_2d * self.gravity * vmask).unsqueeze(0)
        h_in = (self._rho_2d * vmask / (dt * dt)).unsqueeze(0).expand(N, -1, -1)
        H_vv = h_in.unsqueeze(3).unsqueeze(4) * self._eye3
        _t['inertial'] = _t0() - t_

        has_contacts = P > 0 and active_mask.any()
        H_fric = None

        if has_contacts:
            # ---- Contact common ----
            t_ = _t0()
            p_stack = self._p_warm.detach()
            pn_stack = p_normals.detach()
            u_stack = self._u_warm.detach()
            vmask_a = vmask[self._lid_a]                                # [P, mM]
            va = wv_next[:, self._lid_a]                                # [N,P,mM,3]
            va_curr = wv_curr[:, self._lid_a]
            ones = torch.ones(N, P, mM, 1, device=dev, dtype=dty)
            p3 = p_stack[:, :, :3]                                      # [N,P,3]

            va_h = torch.cat([va, ones], 3)                             # [N,P,mM,4]
            d_a = -torch.einsum('npmi,npi->npm', va_h, p_stack)
            val_a, bg_a, bh_a = self._contact_barrier_eval(d_a, x0, self._barrier_d0_half)
            link_mask_a = vmask_a.unsqueeze(0) * amask_f.unsqueeze(2)   # [N,P,mM]
            bg_m = bg_a * link_mask_a
            bh_m = bh_a * link_mask_a
            val_a = val_a * link_mask_a
            norm_p = torch.norm(p3, dim=2)
            val_n, _, _ = barrier_eval(1.0 - norm_p, x0)
            E_contact = coef * (val_a.sum(dim=(1, 2)) + (val_n * amask_f).sum(1))

            if self._gnd_idx.numel() > 0:
                gi = self._gnd_idx
                amask_gnd = amask_f[:, gi]
                d_b_gnd = torch.einsum('gi,npi->npg', self._ground_h, p_stack[:, gi])
                val_b_gnd, _, _ = self._contact_barrier_eval(d_b_gnd, x0, self._barrier_d0_half)
                E_contact = E_contact + coef * (val_b_gnd * amask_gnd.unsqueeze(2)).sum(dim=(1, 2))

            vb_h_lnk, bg_b_lnk_m, bh_b_lnk_m, d_b_lnk, vmask_b = \
                None, None, None, None, None
            bg_b_lnk = None
            if self._lnk_idx.numel() > 0:
                li = self._lnk_idx
                amask_ll = amask_f[:, li]
                lid_b_lnk = self._lid_b[li]
                vb = wv_next[:, lid_b_lnk]
                ones_l = torch.ones(N, li.numel(), mM, 1, device=dev, dtype=dty)
                vb_h_lnk = torch.cat([vb, ones_l], 3)
                vmask_b = vmask[lid_b_lnk]
                d_b_lnk = torch.einsum('npmi,npi->npm', vb_h_lnk, p_stack[:, li])
                val_b_lnk, bg_b_lnk, bh_b_lnk = self._contact_barrier_eval(d_b_lnk, x0, self._barrier_d0_half)
                link_mask_b = vmask_b.unsqueeze(0) * amask_ll.unsqueeze(2)
                bg_b_lnk_m = bg_b_lnk * link_mask_b
                bh_b_lnk_m = bh_b_lnk * link_mask_b
                E_contact = E_contact + coef * (val_b_lnk * link_mask_b).sum(dim=(1, 2))

            E = E + E_contact
            _t['contact_E'] = _t0() - t_

            # ---- Contact g_v, H_vv (vectorised scatter via index_add_) ----
            t_ = _t0()
            g_v_contact_a = -coef * bg_m.unsqueeze(3) * p3.unsqueeze(2)  # [N,P,mM,3]
            # Reshape to [N, P*mM, 3], scatter into [N, L*mM, 3]
            scatter_a_exp = self._scatter_a.unsqueeze(1).expand(-1, mM)  # [P, mM]
            scatter_a_flat = (scatter_a_exp * mM + torch.arange(mM, device=dev).unsqueeze(0)).reshape(-1)  # [P*mM]
            g_v_flat = g_v.reshape(N, L * mM, 3)
            g_v_flat.index_add_(1, scatter_a_flat, g_v_contact_a.reshape(N, P * mM, 3))

            pp = torch.einsum('npi,npj->npij', p3, p3)                  # [N,P,3,3]
            H_contact_a = coef * bh_m.unsqueeze(3).unsqueeze(4) * pp.unsqueeze(2)  # [N,P,mM,3,3]
            H_vv_flat = H_vv.reshape(N, L * mM, 3, 3)
            H_vv_flat.index_add_(1, scatter_a_flat, H_contact_a.reshape(N, P * mM, 3, 3))

            if self._lnk_idx.numel() > 0:
                li = self._lnk_idx
                lid_b_lnk = self._lid_b[li]
                Pl = li.numel()
                p3_lnk = p3[:, li]
                g_v_contact_b = coef * bg_b_lnk_m.unsqueeze(3) * p3_lnk.unsqueeze(2)
                scatter_b_exp = lid_b_lnk.unsqueeze(1).expand(-1, mM)
                scatter_b_flat = (scatter_b_exp * mM + torch.arange(mM, device=dev).unsqueeze(0)).reshape(-1)
                g_v_flat.index_add_(1, scatter_b_flat, g_v_contact_b.reshape(N, Pl * mM, 3))

                pp_lnk = pp[:, li]
                H_contact_b = coef * bh_b_lnk_m.unsqueeze(3).unsqueeze(4) * pp_lnk.unsqueeze(2)
                H_vv_flat.index_add_(1, scatter_b_flat, H_contact_b.reshape(N, Pl * mM, 3, 3))

            g_v = g_v_flat.reshape(N, L, mM, 3)
            H_vv = H_vv_flat.reshape(N, L, mM, 3, 3)
            _t['contact_gH'] = _t0() - t_

            # ---- Friction (matches simulator._assemble_H_theta friction) ----
            t_ = _t0()
            _eps_n = self._friction_eps_n
            _eps_s = self._friction_eps_s
            n_vecs = pn_stack[:, :, :3]
            norm_n = torch.norm(n_vecs, dim=2, keepdim=True) + _eps_n
            n_hat = n_vecs / norm_n
            Proj = self._eye3 - torch.einsum('npi,npj->npij', n_hat, n_hat)

            u_xyz_f, omega_f, t0_fric, t1_fric = \
                self._unified_u_to_xyz_omega_batch(u_stack, n_hat)

            vel_a = (va - va_curr) / dt
            vel = vel_a.clone()
            if self._lnk_idx.numel() > 0:
                li = self._lnk_idx
                lid_b_lnk = self._lid_b[li]
                vb_curr = wv_curr[:, lid_b_lnk]
                vel[:, li] = vel[:, li] - (wv_next[:, lid_b_lnk] - vb_curr) / dt

            tan_vel = torch.einsum('npij,npmj->npmi', Proj, vel)
            r_nx = torch.cross(
                n_hat.unsqueeze(2).expand(-1, -1, mM, -1), va_curr, dim=3)
            omega_term = omega_f.unsqueeze(2).unsqueeze(3) * r_nx
            rel_vel = tan_vel - u_xyz_f.unsqueeze(2) - omega_term

            va_h_fn = torch.cat([va_curr, ones], 3)
            d_fn = -torch.einsum('npmi,npi->npm', va_h_fn, pn_stack)
            _, bg_fn, _ = self._contact_barrier_eval(d_fn, x0, self._barrier_d0_half)
            pn3_f = pn_stack[:, :, :3]
            f_vec_f = coef * bg_fn.unsqueeze(3) * pn3_f.unsqueeze(2)
            fn_sq_f = f_vec_f.pow(2).sum(dim=3)
            sqrt_eps_f = torch.sqrt(torch.tensor(_eps_s, device=dev, dtype=dty))
            A_m = torch.sqrt(fn_sq_f + _eps_s) - sqrt_eps_f

            s_norm = torch.sqrt(rel_vel.pow(2).sum(3) + _eps_s)
            inv_s = 1.0 / (s_norm + 1e-30)
            inv_s3 = inv_s.pow(3)

            E = E + (self.friction * dt * A_m * s_norm * link_mask_a
                     ).sum(dim=(1, 2))

            # Friction g_v
            rel_over_s = rel_vel * inv_s.unsqueeze(3)
            proj_rs = torch.einsum('npij,npmj->npmi', Proj, rel_over_s)
            w_fric = (A_m * link_mask_a).unsqueeze(3)
            fric_gv = self.friction * w_fric * proj_rs
            g_v_flat = g_v.reshape(N, L * mM, 3)
            g_v_flat.index_add_(1, scatter_a_flat, fric_gv.reshape(N, P * mM, 3))
            if self._lnk_idx.numel() > 0:
                g_v_flat.index_add_(1, scatter_b_flat,
                                    -fric_gv[:, li].reshape(N, Pl * mM, 3))
            g_v = g_v_flat.reshape(N, L, mM, 3)

            # Friction H_vv
            weight_hv = self.friction / dt * A_m * link_mask_a
            M_inn = (self._eye3.reshape(1, 1, 1, 3, 3)
                     * inv_s.unsqueeze(3).unsqueeze(4)
                     - torch.einsum('npmi,npmj->npmij', rel_vel, rel_vel)
                       * inv_s3.unsqueeze(3).unsqueeze(4))
            PMP = torch.einsum('npia,npmab,npbj->npmij', Proj, M_inn, Proj)
            H_fric = weight_hv.unsqueeze(3).unsqueeze(4) * PMP
            H_vv_flat = H_vv.reshape(N, L * mM, 3, 3)
            H_vv_flat.index_add_(1, scatter_a_flat,
                                 H_fric.reshape(N, P * mM, 3, 3))
            if self._lnk_idx.numel() > 0:
                H_vv_flat.index_add_(1, scatter_b_flat,
                                     H_fric[:, li].reshape(N, Pl * mM, 3, 3))
            H_vv = H_vv_flat.reshape(N, L, mM, 3, 3)
            _t['friction'] = _t0() - t_

        # ---- g_theta = J^T @ g_v + PD + limits  [N, n] ----
        t_ = _t0()
        B = L * mM
        J2 = J.reshape(N, B, 3, n)
        gv2 = g_v.reshape(N, B, 3)
        g_theta = torch.einsum('nbci,nbc->ni', J2, gv2)

        mask = self.joint_mask
        g_theta = (g_theta
                   + kp * mask.pow(2) * (theta - pd)
                   + kd * mask.pow(2) * (theta - theta_t))

        # Joint-limit barrier: compute once, reuse val/grad/hess for E, g, H
        jl_val_lo = jl_val_hi = jl_bg_lo = jl_bg_hi = jl_bh_lo = jl_bh_hi = None
        if self._use_joint_limit_barrier and self._hinge_idx is not None:
            q = theta[:, self._hinge_idx]
            jl_val_lo, jl_bg_lo, jl_bh_lo = barrier_eval(q - self._hinge_lo, 0.3)
            jl_val_hi, jl_bg_hi, jl_bh_hi = barrier_eval(self._hinge_hi - q, 0.3)
            g_lim = torch.zeros(N, n, device=dev, dtype=dty)
            g_lim[:, self._hinge_idx] = 10.0 * (jl_bg_lo - jl_bg_hi)
            g_theta = g_theta + g_lim

        pos_err = (theta - pd) * mask
        vel_err = (theta - theta_t) * mask
        E = E + 0.5 * kp * pos_err.pow(2).sum(1) + 0.5 * kd * vel_err.pow(2).sum(1)
        if jl_val_lo is not None:
            E = E + (jl_val_lo + jl_val_hi).sum(1) * 10.0
        _t['g_theta'] = _t0() - t_

        # ---- H_theta = J^T H_vv J + cross + FK corr + PD  [N, n, n] ----
        t_ = _t0()
        H2 = H_vv.reshape(N, B, 3, 3)
        JtH = torch.einsum('nbci,nbcd->nbid', J2, H2)
        H_theta = torch.einsum('nbid,nbdj->nij', JtH, J2)

        # Friction cross terms (link-link, vectorised)
        if has_contacts and self._lnk_idx.numel() > 0 and H_fric is not None:
            li = self._lnk_idx
            lid_b_lnk = self._lid_b[li]
            la_lnk = self._lid_a[li]
            Ja = J[:, la_lnk]
            Jb = J[:, lid_b_lnk]
            Hf = H_fric[:, li]
            cross = torch.einsum('npmci,npmcd,npmdj->npij', Ja, Hf, Jb)
            H_theta = H_theta - cross.sum(1) - cross.sum(1).transpose(1, 2)

        H_theta = H_theta + self._compute_fk_correction_batch(
            J, g_v, theta, wv_next, T_all=T_all, dR_cache=dR_cache)

        H_theta = H_theta + torch.diag((kp + kd) * mask * mask).unsqueeze(0)
        if jl_bh_lo is not None:
            diag_lim = torch.zeros(N, n, device=dev, dtype=dty)
            diag_lim[:, self._hinge_idx] = 10.0 * (jl_bh_lo + jl_bh_hi)
            H_theta = H_theta + torch.diag_embed(diag_lim)

        H_theta = 0.5 * (H_theta + H_theta.transpose(1, 2))
        _t['H_theta'] = _t0() - t_

        # ---- Manifold Hessians (stored for Schur) ----
        t_ = _t0()
        if has_contacts:
            self._compute_manifold_hessians_batch(
                J, wv_next, wv_curr, active_mask, p_normals,
                va_h, d_a, bg_a, bh_a, bg_m, bh_m, p3, pp, vmask_a,
                link_mask_a, n_hat, Proj, rel_vel, inv_s, inv_s3, A_m,
                r_nx, s_norm, t0_fric, t1_fric,
                d_b_lnk, bg_b_lnk_m, bh_b_lnk_m, vb_h_lnk, vmask_b,
                bg_b_lnk_raw=bg_b_lnk)
        else:
            self._g_p = torch.zeros(N, P, 4, device=dev, dtype=dty)
            self._H_pp = torch.zeros(N, P, 4, 4, device=dev, dtype=dty)
            self._H_tp = torch.zeros(N, P, n, 4, device=dev, dtype=dty)
            self._g_u = torch.zeros(N, P, 3, device=dev, dtype=dty)
            self._H_uu = torch.zeros(N, P, 3, 3, device=dev, dtype=dty)
            self._H_tu = torch.zeros(N, P, n, 3, device=dev, dtype=dty)
        _t['manifold'] = _t0() - t_

        self._last_timing = _t
        return E.detach(), g_theta.detach(), H_theta.detach()

    # ==================================================================
    #  FK correction (batched) — mirrors _compute_fk_correction_analytic
    # ==================================================================
    def _compute_fk_correction_batch(self, J, g_v, theta, wv,
                                     T_all=None, dR_cache=None):
        dev, dty = self.device, self.dtype
        N, n = theta.shape
        vmask = self._vmask_f
        H_FK = torch.zeros(N, n, n, device=dev, dtype=dty)

        if T_all is None:
            with torch.no_grad():
                T_all = self.robot.forward_kinematics_batch(theta.detach())

        axes_w = [None] * len(self.robot.joints)
        for ji, jt in enumerate(self.robot.joints):
            if jt.jtype != 'hinge':
                continue
            par = jt.parent_link
            R_p = (self._eye3.unsqueeze(0).expand(N, -1, -1)
                   if par < 0 else T_all[:, par, :3, :3])
            a_norm = jt.axis / (jt.axis.norm() + 1e-12)
            axes_w[ji] = torch.einsum('nij,j->ni', R_p, a_norm)

        if dR_cache is None:
            dR_cache = {}

        # ---- Euler (free joint) FK correction — batched over N ----
        for ji, jt in enumerate(self.robot.joints):
            if jt.jtype != 'free':
                continue
            f_off = jt.dof_offset
            R_eu = T_all[:, jt.child_link, :3, :3]

            if ji in dR_cache:
                cr, sr, cp, sp, cy, sy, z, dR_stk = dR_cache[ji]
            else:
                roll  = theta[:, f_off + 3].detach()
                pitch = theta[:, f_off + 4].detach()
                yaw   = theta[:, f_off + 5].detach()
                cr, sr = torch.cos(roll), torch.sin(roll)
                cp, sp = torch.cos(pitch), torch.sin(pitch)
                cy, sy = torch.cos(yaw), torch.sin(yaw)
                z = torch.zeros_like(cr)
                dR_stk = torch.stack([
                    torch.stack([
                        torch.stack([z, cy*sp*cr+sy*sr, -cy*sp*sr+sy*cr], 1),
                        torch.stack([z, sy*sp*cr-cy*sr, -sy*sp*sr-cy*cr], 1),
                        torch.stack([z, cp*cr, -cp*sr], 1)], 1),
                    torch.stack([
                        torch.stack([-cy*sp, cy*cp*sr, cy*cp*cr], 1),
                        torch.stack([-sy*sp, sy*cp*sr, sy*cp*cr], 1),
                        torch.stack([-cp, -sp*sr, -sp*cr], 1)], 1),
                    torch.stack([
                        torch.stack([-sy*cp, -sy*sp*sr-cy*cr, -sy*sp*cr+cy*sr], 1),
                        torch.stack([cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], 1),
                        torch.stack([z, z, z], 1)], 1)], 0).permute(1, 0, 2, 3)

            M_stk = torch.einsum('nrij,nkj->nrik', dR_stk, R_eu)

            # Euler-Euler (vectorised: no idx_map loop)
            t_eu = T_all[:, jt.child_link, :3, 3]
            diff = wv - t_eu.reshape(N, 1, 1, 3)
            q_all = torch.einsum('nij,nlmj->nlmi', R_eu.transpose(1,2), diff)
            g_fl = (g_v * vmask.unsqueeze(0).unsqueeze(3)).reshape(N, -1, 3)
            q_fl = (q_all * vmask.unsqueeze(0).unsqueeze(3)).reshape(N, -1, 3)
            Q = torch.einsum('nbi,nbj->nij', g_fl, q_fl)

            d2R = self._compute_d2R_batch(cr, sr, cp, sp, cy, sy, z)
            vals_ee = (d2R * Q.unsqueeze(0)).sum(dim=(2, 3))  # [6, N]
            ri = torch.tensor([0, 0, 0, 1, 1, 2], device=dev, dtype=torch.long)
            si = torch.tensor([0, 1, 2, 1, 2, 2], device=dev, dtype=torch.long)
            fo = f_off + 3
            H_FK[:, fo + ri, fo + si] = vals_ee.T
            H_FK[:, fo + si, fo + ri] = vals_ee.T

            # Euler-Hinge (vectorised inner r-loop)
            for ki, jt_k in enumerate(self.robot.joints):
                if jt_k.jtype != 'hinge':
                    continue
                k_off = jt_k.dof_offset
                desc_k = self._descendants[ki]
                if not desc_k:
                    continue
                dt_ = self._desc_tensors[ki]
                J_k = J[:, dt_, :, :, k_off]
                g_d = g_v[:, dt_]
                mk = vmask[dt_].unsqueeze(0).unsqueeze(3)
                g_f = (g_d * mk).reshape(N, -1, 3)
                j_f = (J_k * mk).reshape(N, -1, 3)
                W_k = torch.einsum('nbi,nbj->nij', g_f, j_f)
                vals_eh = (M_stk * W_k.unsqueeze(1)).sum(dim=(2, 3))  # [N, 3]
                H_FK[:, fo:fo+3, k_off] = vals_eh
                H_FK[:, k_off, fo:fo+3] = vals_eh

            break

        # ---- Hinge-Hinge FK correction ----
        for ki, jt_k in enumerate(self.robot.joints):
            if jt_k.jtype != 'hinge':
                continue
            k_off = jt_k.dof_offset
            desc_k = self._descendants[ki]
            if not desc_k:
                continue
            dt_ = self._desc_tensors[ki]
            J_k = J[:, dt_, :, :, k_off]
            g_d = g_v[:, dt_]
            mk = vmask[dt_].unsqueeze(0).unsqueeze(3)
            S_k = (torch.linalg.cross(J_k, g_d) * mk).sum(dim=(1, 2))

            H_FK[:, k_off, k_off] = (axes_w[ki] * S_k).sum(1)

            for ji in self._ancestors[ki]:
                jt_j = self.robot.joints[ji]
                j_off = jt_j.dof_offset
                if jt_j.jtype == 'hinge':
                    val = (axes_w[ji] * S_k).sum(1)
                    H_FK[:, j_off, k_off] = val
                    H_FK[:, k_off, j_off] = val

        return H_FK

    @staticmethod
    def _compute_d2R_batch(cr, sr, cp, sp, cy, sy, z):
        """Batched d²R: inputs [N], returns [6, N, 3, 3] for upper-tri (r,s)."""
        def s3(a, b, c):
            return torch.stack([a, b, c], dim=1)
        def m3(r0, r1, r2):
            return torch.stack([r0, r1, r2], dim=1)
        d00 = m3(s3(z, -cy*sp*sr+sy*cr, -cy*sp*cr-sy*sr),
                 s3(z, -sy*sp*sr-cy*cr, -sy*sp*cr+cy*sr),
                 s3(z, -cp*sr, -cp*cr))
        d01 = m3(s3(z, cy*cp*cr, -cy*cp*sr),
                 s3(z, sy*cp*cr, -sy*cp*sr),
                 s3(z, -sp*cr, sp*sr))
        d02 = m3(s3(z, -sy*sp*cr+cy*sr, sy*sp*sr+cy*cr),
                 s3(z, cy*sp*cr+sy*sr, -cy*sp*sr+sy*cr),
                 s3(z, z, z))
        d11 = m3(s3(-cy*cp, -cy*sp*sr, -cy*sp*cr),
                 s3(-sy*cp, -sy*sp*sr, -sy*sp*cr),
                 s3(sp, -cp*sr, -cp*cr))
        d12 = m3(s3(sy*sp, -sy*cp*sr, -sy*cp*cr),
                 s3(-cy*sp, cy*cp*sr, cy*cp*cr),
                 s3(z, z, z))
        d22 = m3(s3(-cy*cp, -cy*sp*sr+sy*cr, -cy*sp*cr-sy*sr),
                 s3(-sy*cp, -sy*sp*sr-cy*cr, -sy*sp*cr+cy*sr),
                 s3(z, z, z))
        return torch.stack([d00, d01, d02, d11, d12, d22], dim=0)  # [6,N,3,3]

    # ==================================================================
    #  Manifold Hessians  (H_pp, H_uu, H_tp, H_tu, g_p, g_u)
    #  Mirrors simulator._compute_manifold_hessians exactly, batched.
    # ==================================================================
    def _compute_manifold_hessians_batch(
            self, J, wv_next, wv_curr, active_mask, p_normals,
            va_h, d_a, bg_a, bh_a, bg_m, bh_m, p3, pp, vmask_a,
            link_mask_a, n_hat, Proj, rel_vel, inv_s, inv_s3, A_m,
            r_nx, s_norm, t0_fric, t1_fric,
            d_b_lnk, bg_b_lnk_m, bh_b_lnk_m, vb_h_lnk, vmask_b,
            bg_b_lnk_raw=None):
        dev, dty = self.device, self.dtype
        N, n = self.N, self._n
        P, mM = self._P, self._max_M
        coef, x0 = self.coef_barrier, self.x0
        dt = self.dt
        vmask = self._vmask_f
        amask_f = active_mask.to(dty)
        _eps_n = self._friction_eps_n

        # ---- H_pp [N, P, 4, 4] ----
        H_pp_a = torch.einsum('npm,npmi,npmj->npij', bh_m, va_h, va_h)

        H_pp_b = torch.zeros(N, P, 4, 4, device=dev, dtype=dty)
        bg_b_gnd = None
        bh_b_gnd = None
        if self._gnd_idx.numel() > 0:
            gi = self._gnd_idx
            amask_gnd = amask_f[:, gi]
            d_b_gnd_v = torch.einsum('gi,npi->npg', self._ground_h,
                                     self._p_warm[:, gi])
            _, bg_b_gnd, bh_b_gnd = self._contact_barrier_eval(
                d_b_gnd_v, x0, self._barrier_d0_half)
            H_pp_b[:, gi] = torch.einsum(
                'npg,gi,gj->npij', bh_b_gnd * amask_gnd.unsqueeze(2),
                self._ground_h, self._ground_h)

        if self._lnk_idx.numel() > 0 and bh_b_lnk_m is not None:
            li = self._lnk_idx
            H_pp_b[:, li] = torch.einsum(
                'npm,npmi,npmj->npij', bh_b_lnk_m, vb_h_lnk, vb_h_lnk)

        np3 = torch.norm(p3, dim=2, keepdim=True) + _eps_n
        n_hat_p = p3 / np3
        s_p = 1.0 - np3.squeeze(2)
        _, bg_n, bh_n = barrier_eval(s_p, x0)
        nn_p = torch.einsum('npi,npj->npij', n_hat_p, n_hat_p)
        I_nn_p = self._eye3 - nn_p
        H_pp_n33 = (bh_n.unsqueeze(2).unsqueeze(3) * nn_p
                     - bg_n.unsqueeze(2).unsqueeze(3) * I_nn_p
                       / np3.unsqueeze(3))
        H_pp_n = torch.zeros(N, P, 4, 4, device=dev, dtype=dty)
        H_pp_n[:, :, :3, :3] = H_pp_n33 * amask_f.unsqueeze(2).unsqueeze(3)

        self._H_pp = coef * (H_pp_a + H_pp_b + H_pp_n)
        self._H_pp = 0.5 * (self._H_pp + self._H_pp.transpose(2, 3))

        # ---- g_p [N, P, 4] ----
        g_p_a = -coef * torch.einsum('npm,npmi->npi', bg_m, va_h)
        g_p_b = torch.zeros(N, P, 4, device=dev, dtype=dty)
        if self._gnd_idx.numel() > 0 and bg_b_gnd is not None:
            gi = self._gnd_idx
            amask_gnd = amask_f[:, gi]
            g_p_b[:, gi] = coef * torch.einsum(
                'npg,gi->npi', bg_b_gnd * amask_gnd.unsqueeze(2),
                self._ground_h)
        if self._lnk_idx.numel() > 0 and bg_b_lnk_m is not None:
            li = self._lnk_idx
            g_p_b[:, li] = coef * torch.einsum(
                'npm,npmi->npi', bg_b_lnk_m, vb_h_lnk)
        g_p_n = torch.zeros(N, P, 4, device=dev, dtype=dty)
        g_p_n[:, :, :3] = (-coef * bg_n.unsqueeze(2) * n_hat_p
                           * amask_f.unsqueeze(2))
        self._g_p = g_p_a + g_p_b + g_p_n

        # ---- H_tp [N, P, n, 4] ----
        J_a_all = J[:, self._lid_a]
        cross_1 = torch.einsum('npi,npmj,npm->npmij', p3, va_h, bh_m)
        cross_2 = torch.zeros(N, P, mM, 3, 4, device=dev, dtype=dty)
        cross_2[:, :, :, :3, :3] = (-bg_m.unsqueeze(3).unsqueeze(4)
                                    * self._eye3)
        cross_a = coef * (cross_1 + cross_2)
        self._H_tp = torch.einsum('npmci,npmcj->npij', J_a_all, cross_a)

        if self._lnk_idx.numel() > 0 and bh_b_lnk_m is not None:
            li = self._lnk_idx
            lid_b_lnk = self._lid_b[li]
            J_b_lnk = J[:, lid_b_lnk]
            p3_lnk = p3[:, li]
            cross_b1 = torch.einsum('npi,npmj,npm->npmij',
                                    p3_lnk, vb_h_lnk, bh_b_lnk_m)
            cross_b2 = torch.zeros(N, li.numel(), mM, 3, 4,
                                   device=dev, dtype=dty)
            vmask_b_all = vmask_b
            amask_ll = amask_f[:, li]
            if bg_b_lnk_raw is not None:
                bg_b2_m = (bg_b_lnk_raw * vmask_b_all.unsqueeze(0)
                           * amask_ll.unsqueeze(2))
            else:
                _, bg_b_lnk2, _ = self._contact_barrier_eval(
                    d_b_lnk, x0, self._barrier_d0_half)
                bg_b2_m = (bg_b_lnk2 * vmask_b_all.unsqueeze(0)
                           * amask_ll.unsqueeze(2))
            cross_b2[:, :, :, :3, :3] = (bg_b2_m.unsqueeze(3).unsqueeze(4)
                                         * self._eye3)
            cross_b = coef * (cross_b1 + cross_b2)
            H_tp_b = torch.einsum('npmci,npmcj->npij', J_b_lnk, cross_b)
            self._H_tp[:, li] = self._H_tp[:, li] + H_tp_b

        # ---- Friction: H_uu, g_u, H_tu  (with tangent-basis lift/reduce,
        #      matching simulator._compute_manifold_hessians) ----
        if self._use_friction:
            c_mh = self.friction * dt * A_m * link_mask_a        # [N,P,mM]

            M_inn = (self._eye3.reshape(1, 1, 1, 3, 3)
                     * inv_s.unsqueeze(3).unsqueeze(4)
                     - torch.einsum('npmi,npmj->npmij', rel_vel, rel_vel)
                       * inv_s3.unsqueeze(3).unsqueeze(4))
            H_uu_33 = torch.einsum('npm,npmij->npij', c_mh, M_inn)

            h_mh = (rel_vel * r_nx).sum(dim=3)                  # [N,P,mM]
            t_mix = (r_nx / s_norm.unsqueeze(3)
                     - (h_mh / s_norm.pow(2)).unsqueeze(3)
                       * rel_vel / s_norm.unsqueeze(3))          # [N,P,mM,3]
            col_mh = torch.einsum('npm,npmc->npc', c_mh, t_mix) # [N,P,3]
            rnx_sq = r_nx.pow(2).sum(dim=3)                      # [N,P,mM]
            H_ww = torch.einsum(
                'npm,npm->np', c_mh,
                rnx_sq / s_norm - h_mh.pow(2) / s_norm.pow(3))  # [N,P]

            H_uu_4 = torch.zeros(N, P, 4, 4, device=dev, dtype=dty)
            H_uu_4[:, :, :3, :3] = 0.5 * (H_uu_33
                                           + H_uu_33.transpose(2, 3))
            H_uu_4[:, :, :3, 3] = col_mh
            H_uu_4[:, :, 3, :3] = col_mh
            H_uu_4[:, :, 3, 3] = H_ww
            H_uu_4 = 0.5 * (H_uu_4 + H_uu_4.transpose(2, 3))
            J_lift = self._friction_J_lift_batch(t0_fric, t1_fric)
            self._H_uu = self._friction_reduce_H_batch(H_uu_4, J_lift)
            self._H_uu = 0.5 * (self._H_uu + self._H_uu.transpose(2, 3))

            # g_u [N, P, 3]
            rel_over_s = rel_vel * inv_s.unsqueeze(3)
            g_u3 = -torch.einsum('npm,npmc->npc', c_mh, rel_over_s)
            g_om = -(c_mh * h_mh / s_norm).sum(dim=2)           # [N,P]
            self._g_u = self._friction_reduce_g_batch(g_u3, g_om, J_lift)

            # H_tu [N, P, n, 4] → reduced to [N, P, n, 3]
            Proj_rel = torch.einsum('npij,npmj->npmi', Proj, rel_vel)
            M_wu = (Proj.unsqueeze(2) * inv_s.unsqueeze(3).unsqueeze(4)
                    - torch.einsum('npmi,npmj->npmij', Proj_rel, rel_vel)
                      * inv_s3.unsqueeze(3).unsqueeze(4))
            w_wu = -(self.friction * A_m * link_mask_a)
            cross_wu = w_wu.unsqueeze(3).unsqueeze(4) * M_wu

            H_tu_u = torch.einsum('npmci,npmcj->npij',
                                  J_a_all, cross_wu)             # [N,P,n,3]
            if self._lnk_idx.numel() > 0:
                li = self._lnk_idx
                lid_b_lnk = self._lid_b[li]
                J_b_lnk = J[:, lid_b_lnk]
                H_tu_u[:, li] = H_tu_u[:, li] + torch.einsum(
                    'npmci,npmcj->npij', J_b_lnk, -cross_wu[:, li])

            inner_om = (-r_nx / s_norm.unsqueeze(3)
                        + (h_mh / s_norm.pow(2)).unsqueeze(3)
                          * rel_vel / s_norm.unsqueeze(3))
            cross_wom = (self.friction
                         * (A_m * link_mask_a).unsqueeze(3)
                         * torch.einsum('npij,npmj->npmi',
                                        Proj, inner_om))
            H_tu_om = torch.einsum('npmci,npmc->npi',
                                   J_a_all, cross_wom)           # [N,P,n]
            if self._lnk_idx.numel() > 0:
                li = self._lnk_idx
                lid_b_lnk = self._lid_b[li]
                J_b_lnk = J[:, lid_b_lnk]
                H_tu_om[:, li] = H_tu_om[:, li] - torch.einsum(
                    'npmci,npmc->npi', J_b_lnk, cross_wom[:, li])

            H_tu_4 = torch.cat([H_tu_u, H_tu_om.unsqueeze(3)],
                               dim=3)                            # [N,P,n,4]
            self._H_tu = self._friction_reduce_Htu_batch(
                H_tu_4, J_lift)                                  # [N,P,n,3]
        else:
            self._H_uu = torch.zeros(N, P, 3, 3, device=dev, dtype=dty)
            self._g_u = torch.zeros(N, P, 3, device=dev, dtype=dty)
            self._H_tu = torch.zeros(N, P, n, 3, device=dev, dtype=dty)

    # ==================================================================
    #  Schur update  (strict mirror of simulator.py _schur_update,
    #                  batched over N envs)
    #
    #  Single-env reference (simulator.py):
    #    g_bar = g_theta.clone()
    #    H_bar = H_theta.clone()
    #    # eliminate p:
    #    Hpp_reg = H_pp + alpha*lm_g*I4          # [K,4,4]
    #    sol     = solve(Hpp_reg, [g_p | Htp^T])  # [K,4,1+n]
    #    inv_gp  = sol[:,:,0]                      # [K,4]
    #    inv_Htp = sol[:,:,1:]                     # [K,4,n]
    #    g_bar  -= einsum('kni,ki->n', Htp, inv_gp)
    #    H_bar  -= einsum('kni,kij->nj', Htp, inv_Htp)
    #    # eliminate u: same pattern with H_uu/g_u/H_tu
    #    H_bar  += alpha * I_n
    #    H_bar   = 0.5*(H_bar + H_bar^T)
    #    delta   = -solve(H_bar, g_bar)
    #
    #  Batch version adds leading dim N; inactive manifolds have
    #  H_pp=0 / g_p=0 / H_tp=0, so their Schur contribution is zero
    #  after masking by active_mask.
    # ==================================================================
    def _schur_update_batch(self, g_theta, H_theta, active_mask, alpha):
        dev, dty = self.device, self.dtype
        N, n = g_theta.shape
        P = self._P
        lm_g = self.lm_gamma
        NP = N * P
        amask_f = active_mask.to(dty)                      # [N, P]

        g_bar = g_theta.clone()                             # [N, n]
        H_bar = H_theta.clone()                             # [N, n, n]

        # --- Eliminate p ---
        # Hpp_reg[n,k] = H_pp[n,k] + alpha[n]*lm_g*I4     [N,P,4,4]
        Hpp_reg = (self._H_pp
                   + (alpha[:, None, None, None] * lm_g) * self._eye4)
        gp  = self._g_p                                    # [N,P,4]
        Htp = self._H_tp                                   # [N,P,n,4]

        # Batched RHS = [g_p | Htp^T]                      [N,P,4,1+n]
        rhs_p = torch.cat([gp.unsqueeze(3),
                           Htp.transpose(2, 3)], dim=3)
        # solve  Hpp_reg @ sol = rhs  →  sol               [N,P,4,1+n]
        sol_p = torch.linalg.solve(
            Hpp_reg.reshape(NP, 4, 4),
            rhs_p.reshape(NP, 4, 1 + n)
        ).reshape(N, P, 4, 1 + n)
        inv_gp  = sol_p[:, :, :, 0]                        # [N,P,4]
        inv_Htp = sol_p[:, :, :, 1:]                       # [N,P,4,n]

        # g_bar -= Σ_k  Htp[k]  @  inv_gp[k]
        #   single-env: einsum('kni,ki->n', Htp, inv_gp)
        #   batch:      einsum('npni,npi->nn') → per-env sum over P
        g_bar = g_bar - (amask_f.unsqueeze(2)
                         * torch.einsum('npji,npi->npj', Htp, inv_gp)
                         ).sum(1)
        # H_bar -= Σ_k  Htp[k]  @  inv_Htp[k]
        #   single-env: einsum('kni,kij->nj', Htp, inv_Htp)
        H_bar = H_bar - (amask_f.unsqueeze(2).unsqueeze(3)
                         * torch.einsum('npji,npik->npjk', Htp, inv_Htp)
                         ).sum(1)

        # --- Eliminate u ---
        Huu_reg = (self._H_uu
                   + (alpha[:, None, None, None] * lm_g) * self._eye3)
        gu  = self._g_u                                    # [N,P,3]
        Htu = self._H_tu                                   # [N,P,n,3]

        rhs_u = torch.cat([gu.unsqueeze(3),
                           Htu.transpose(2, 3)], dim=3)    # [N,P,3,1+n]
        sol_u = torch.linalg.solve(
            Huu_reg.reshape(NP, 3, 3),
            rhs_u.reshape(NP, 3, 1 + n)
        ).reshape(N, P, 3, 1 + n)
        inv_gu  = sol_u[:, :, :, 0]                        # [N,P,3]
        inv_Htu = sol_u[:, :, :, 1:]                       # [N,P,3,n]

        g_bar = g_bar - (amask_f.unsqueeze(2)
                         * torch.einsum('npji,npi->npj', Htu, inv_gu)
                         ).sum(1)
        H_bar = H_bar - (amask_f.unsqueeze(2).unsqueeze(3)
                         * torch.einsum('npji,npik->npjk', Htu, inv_Htu)
                         ).sum(1)

        # --- LM damping + symmetrise ---
        H_bar = H_bar + alpha.unsqueeze(1).unsqueeze(2) * self._eye_n
        H_bar = 0.5 * (H_bar + H_bar.transpose(1, 2))

        # --- solve for delta ---
        try:
            delta_theta = -torch.linalg.solve(H_bar, g_bar)
        except Exception:
            delta_theta = -g_bar * 0.01

        return delta_theta, g_bar.detach(), H_bar.detach()

    def _solve_theta_direct_batch(self, g_theta, H_theta, alpha):
        """Batch ``(H_theta + alpha * I) delta = -g_theta`` without Schur."""
        H_reg = H_theta + alpha.unsqueeze(1).unsqueeze(2) * self._eye_n
        H_reg = 0.5 * (H_reg + H_reg.transpose(1, 2))
        try:
            delta = -torch.linalg.solve(H_reg, g_theta)
        except Exception:
            delta = -g_theta * 0.01
        return delta, H_reg.detach()

    # ==================================================================
    #  Back-substitution  (strict mirror of simulator.py _back_substitute,
    #                       batched over N envs)
    #
    #  Single-env reference:
    #    rhs_p = g_p + einsum('kni,n->ki', Htp, delta_theta)
    #    dp    = solve(Hpp_reg, rhs_p)
    #    p    -= dp
    #    rhs_u = g_u + einsum('kni,n->ki', Htu, delta_theta)
    #    du    = solve(Huu_reg, rhs_u)
    #    u    -= du
    # ==================================================================
    def _back_substitute_batch(self, p_trial, u_trial, delta_theta,
                               active_mask, alpha):
        dev, dty = self.device, self.dtype
        N, n = delta_theta.shape
        P = self._P
        lm_g = self.lm_gamma
        NP = N * P
        amask = active_mask.unsqueeze(2).to(dty)            # [N,P,1]

        # --- p update ---
        Hpp_reg = (self._H_pp
                   + (alpha[:, None, None, None] * lm_g) * self._eye4)
        # rhs = g_p + Htp^T @ delta_theta
        #   single-env einsum('kni,n->ki', Htp, delta)
        #   Htp is [N,P,n,4], delta is [N,n]
        rhs_p = self._g_p + torch.einsum('npji,nj->npi',
                                         self._H_tp, delta_theta)
        dp = torch.linalg.solve(
            Hpp_reg.reshape(NP, 4, 4),
            rhs_p.reshape(NP, 4)
        ).reshape(N, P, 4)
        p_trial[:] = p_trial - dp * amask

        # --- u update ---
        Huu_reg = (self._H_uu
                   + (alpha[:, None, None, None] * lm_g) * self._eye3)
        rhs_u = self._g_u + torch.einsum('npji,nj->npi',
                                         self._H_tu, delta_theta)
        du = torch.linalg.solve(
            Huu_reg.reshape(NP, 3, 3),
            rhs_u.reshape(NP, 3)
        ).reshape(N, P, 3)
        u_trial[:] = u_trial - du * amask

    # ==================================================================
    #  LM console (same style as :meth:`simulator.ConvexHullPBADSimulator._solve_lm`),
    #  one line per environment tagged ``[LM env=k]``.
    # ==================================================================
    def _output_lm_headers(
            self, E: torch.Tensor, g: torch.Tensor, alpha: torch.Tensor,
            active_mask: torch.Tensor,
            which: Optional[torch.Tensor] = None) -> None:
        if not self._output:
            return
        mode = "Schur" if self._implicit else "explicit"
        for e in range(self.N):
            if which is not None and not bool(which[e].item()):
                continue
            K = int(active_mask[e].sum().item())
            print(
                f"  [LM env={e}] contacts={K}  E0={float(E[e]):.6e}  "
                f"max|g_θ|={float(g[e].abs().max()):.3e}  "
                f"alpha={float(alpha[e]):.2e}  ({mode})",
                flush=True)

    def _output_lm_reject_fail_all(
            self, live: torch.Tensor, rej: torch.Tensor, acc: torch.Tensor,
            E_try: torch.Tensor, alpha: torch.Tensor, nu: torch.Tensor,
            penetrating: torch.Tensor, alpha_max: float) -> None:
        if not self._output:
            return
        thr = self._reject_step_energy_above
        thr_f = float(thr) if thr is not None else None
        for e in range(self.N):
            if not bool(live[e].item()):
                continue
            if bool(acc[e].item()) or not bool(rej[e].item()):
                continue
            E_e = float(E_try[e].item())
            a_e = float(alpha[e].item())
            nu_e = float(nu[e].item())
            pen_e = bool(penetrating[e].item())
            tag = "  [REJECT]"
            if not math.isfinite(E_e):
                tag = "  [REJECT non-finite E]"
            elif thr_f is not None and math.isfinite(E_e) and E_e > thr_f:
                tag = f"  [REJECT bad_state: E_trial>{thr_f:g}]"
            elif pen_e:
                tag = "  [REJECT penetration]"
            print(
                f"    [LM env={e}] attempt: E_trial={E_e:.6e}  alpha={a_e:.2e}  "
                f"nu={nu_e:.0f}{tag}",
                flush=True)
            if a_e >= alpha_max:
                print(
                    f"    [LM env={e}] FAIL: alpha >= alpha_max",
                    flush=True)

    def _output_lm_accept_converged_all(
            self, acc: torch.Tensor, it_accepted: torch.Tensor,
            E: torch.Tensor, g: torch.Tensor, alpha: torch.Tensor,
            rho: torch.Tensor, active_mask: torch.Tensor,
            g_row: torch.Tensor) -> None:
        if not self._output or not acc.any():
            return
        for e in range(self.N):
            if not bool(acc[e].item()):
                continue
            K = int(active_mask[e].sum().item())
            rho_e = float(rho[e].item())
            print(
                f"    [LM env={e}] iter {int(it_accepted[e].item()):2d}: "
                f"E={float(E[e].item()):.6e}  "
                f"max|g_θ|={float(g[e].abs().max().item()):.3e}  "
                f"alpha={float(alpha[e].item()):.2e}  rho={rho_e:.4f}  "
                f"K={K}  [ACCEPT]",
                flush=True)
            if float(g_row[e].item()) < self.gtol:
                print(
                    f"    [LM env={e}] converged: max|g_θ|="
                    f"{float(g_row[e].item()):.2e} < gtol={self.gtol:.1e}",
                    flush=True)

    # ==================================================================
    #  Friction cross-Hessian  ∂²E_fric/(∂θ_{t+1} ∂θ_t)   [N, n, n]
    #  Dominant velocity-chain contribution (matches simulator._friction_cross_theta_t)
    # ==================================================================
    def _friction_cross_theta_t_batch(self, wv_star, wv_curr,
                                      J_star, J_t, active_mask,
                                      friction_pn):
        dev, dty = self.device, self.dtype
        N, n = self.N, self._n
        P, mM = self._P, self._max_M
        dt = self.dt

        if P == 0 or not self._use_friction:
            return torch.zeros(N, n, n, device=dev, dtype=dty)

        lid_a = self._lid_a
        lid_b = self._lid_b
        li = self._lnk_idx
        amask_f = active_mask.to(dty)

        vmask_a = self._vmask_f[lid_a]
        va_next = wv_star[:, lid_a]
        va_curr = wv_curr[:, lid_a]

        _eps_n = self._friction_eps_n
        _eps_s = self._friction_eps_s
        n_vecs = friction_pn[:, :, :3]
        norm_n = torch.norm(n_vecs, dim=2, keepdim=True) + _eps_n
        n_hat = n_vecs / norm_n
        Proj = (self._eye3.reshape(1, 1, 3, 3)
                - torch.einsum('npi,npj->npij', n_hat, n_hat))

        vel = (va_next - va_curr) / dt
        if li.numel() > 0:
            lid_b_lnk = lid_b[li]
            vel[:, li] = (vel[:, li]
                          - (wv_star[:, lid_b_lnk]
                             - wv_curr[:, lid_b_lnk]) / dt)
        tan_vel = torch.einsum('npij,npmj->npmi', Proj, vel)

        u_stack = self._u_warm.detach()
        u_xyz, omega_s, _, _ = self._unified_u_to_xyz_omega_batch(
            u_stack, n_hat)
        r_nx = torch.cross(
            n_hat.unsqueeze(2).expand_as(va_curr), va_curr, dim=3)
        rel_vel = (tan_vel - u_xyz.unsqueeze(2)
                   - omega_s.unsqueeze(2).unsqueeze(3) * r_nx)

        coef = self.coef_barrier
        ones_1 = torch.ones(N, P, mM, 1, device=dev, dtype=dty)
        va_h = torch.cat([va_curr, ones_1], dim=3)
        d_fn = -torch.einsum('npmi,npi->npm', va_h, friction_pn)
        _, bg_fn, _ = self._contact_barrier_eval(
            d_fn, self.x0, self._barrier_d0_half)
        pn3 = friction_pn[:, :, :3]
        f_vec = coef * bg_fn.unsqueeze(3) * pn3.unsqueeze(2)
        fn_sq = f_vec.pow(2).sum(dim=3)
        sqrt_eps = torch.sqrt(torch.tensor(_eps_s, device=dev, dtype=dty))
        A_m = torch.sqrt(fn_sq + _eps_s) - sqrt_eps

        s_norm = torch.sqrt(rel_vel.pow(2).sum(3) + _eps_s)
        inv_s = 1.0 / (s_norm + 1e-30)
        inv_s3 = inv_s.pow(3)

        weight = (self.friction / dt * A_m
                  * vmask_a.unsqueeze(0) * amask_f.unsqueeze(2))
        M_inn = (self._eye3.reshape(1, 1, 1, 3, 3)
                 * inv_s.unsqueeze(3).unsqueeze(4)
                 - torch.einsum('npmi,npmj->npmij', rel_vel, rel_vel)
                   * inv_s3.unsqueeze(3).unsqueeze(4))
        PMP = torch.einsum('npia,npmab,npbj->npmij', Proj, M_inn, Proj)
        H_fric = weight.unsqueeze(3).unsqueeze(4) * PMP

        J_s_a = J_star[:, lid_a]
        J_t_a = J_t[:, lid_a]

        Hx = -torch.einsum('npmci,npmcd,npmdj->nij', J_s_a, H_fric, J_t_a)

        if li.numel() > 0:
            lid_b_lnk = lid_b[li]
            J_s_b = J_star[:, lid_b_lnk]
            J_t_b = J_t[:, lid_b_lnk]
            Hf_ll = H_fric[:, li]
            Js_a_ll = J_s_a[:, li]
            Jt_a_ll = J_t_a[:, li]
            Hx = Hx - torch.einsum(
                'npmci,npmcd,npmdj->nij', J_s_b, Hf_ll, J_t_b)
            Hx = Hx + torch.einsum(
                'npmci,npmcd,npmdj->nij', Js_a_ll, Hf_ll, J_t_b)
            Hx = Hx + torch.einsum(
                'npmci,npmcd,npmdj->nij', J_s_b, Hf_ll, Jt_a_ll)

        return Hx

    # ==================================================================
    #  ∂g_u/∂θ_t  (batched)  →  [N, P, 3, n]
    #  Mirrors simulator._compute_dgu_dtheta_t, batched over N envs.
    # ==================================================================
    def _compute_dgu_dtheta_t_batch(self, wv_star, wv_curr,
                                    J_t, active_mask, friction_pn):
        dev, dty = self.device, self.dtype
        N, n = self.N, self._n
        P, mM = self._P, self._max_M
        dt = self.dt

        if P == 0 or not self._use_friction:
            return None

        lid_a = self._lid_a
        lid_b = self._lid_b
        li = self._lnk_idx
        gi = self._gnd_idx
        amask_f = active_mask.to(dty)

        vmask_a = self._vmask_f[lid_a]
        va_next = wv_star[:, lid_a]
        va_curr = wv_curr[:, lid_a]

        _eps_n = self._friction_eps_n
        _eps_s = self._friction_eps_s
        n_vecs = friction_pn[:, :, :3]
        norm_n = torch.norm(n_vecs, dim=2, keepdim=True) + _eps_n
        n_hat = n_vecs / norm_n
        Proj = (self._eye3.reshape(1, 1, 3, 3)
                - torch.einsum('npi,npj->npij', n_hat, n_hat))

        vel = (va_next - va_curr) / dt
        lid_b_lnk = None
        if li.numel() > 0:
            lid_b_lnk = lid_b[li]
            vel[:, li] = (vel[:, li]
                          - (wv_star[:, lid_b_lnk]
                             - wv_curr[:, lid_b_lnk]) / dt)
        tan_vel = torch.einsum('npij,npmj->npmi', Proj, vel)

        u_stack = self._u_warm.detach()
        u_xyz, omega_s, t0_dgu, t1_dgu = \
            self._unified_u_to_xyz_omega_batch(u_stack, n_hat)
        r_nx = torch.cross(
            n_hat.unsqueeze(2).expand_as(va_curr), va_curr, dim=3)
        rel_vel = (tan_vel - u_xyz.unsqueeze(2)
                   - omega_s.unsqueeze(2).unsqueeze(3) * r_nx)

        coef = self.coef_barrier
        ones_1 = torch.ones(N, P, mM, 1, device=dev, dtype=dty)
        va_h = torch.cat([va_curr, ones_1], dim=3)
        d_fn = -torch.einsum('npmi,npi->npm', va_h, friction_pn)
        _, bg_fn, _ = self._contact_barrier_eval(
            d_fn, self.x0, self._barrier_d0_half)
        pn3 = friction_pn[:, :, :3]
        f_vec = coef * bg_fn.unsqueeze(3) * pn3.unsqueeze(2)
        fn_sq = f_vec.pow(2).sum(dim=3)
        sqrt_eps = torch.sqrt(torch.tensor(_eps_s, device=dev, dtype=dty))
        A_m = torch.sqrt(fn_sq + _eps_s) - sqrt_eps

        s_norm = torch.sqrt(rel_vel.pow(2).sum(3) + _eps_s)
        inv_s = 1.0 / (s_norm + 1e-30)
        inv_s3 = inv_s.pow(3)

        Proj_rel = torch.einsum('npij,npmj->npmi', Proj, rel_vel)
        MP = (Proj.unsqueeze(2) * inv_s.unsqueeze(3).unsqueeze(4)
              - torch.einsum('npmi,npmj->npmij', rel_vel, Proj_rel)
                * inv_s3.unsqueeze(3).unsqueeze(4))

        J_t_a = J_t[:, lid_a]
        J_t_eff = J_t_a.clone()
        if li.numel() > 0:
            J_t_eff[:, li] = J_t_eff[:, li] - J_t[:, lid_b_lnk]

        c_vel = (self.friction * A_m
                 * vmask_a.unsqueeze(0) * amask_f.unsqueeze(2))
        c_fm = (self.friction * dt * A_m
                * vmask_a.unsqueeze(0) * amask_f.unsqueeze(2))
        dgu_3 = torch.einsum('npm,npmcd,npmdj->npcj', c_vel, MP, J_t_eff)

        # --- omega component ---
        h_dgu = (rel_vel * r_nx).sum(dim=3)
        omv = omega_s.unsqueeze(2).unsqueeze(3)
        Proj_rnx = torch.einsum('npij,npmj->npmi', Proj, r_nx)
        grad_h = (-Proj_rnx / dt
                  + torch.cross(rel_vel,
                                n_hat.unsqueeze(2).expand_as(rel_vel),
                                dim=3)
                  + omv * torch.cross(
                      n_hat.unsqueeze(2).expand_as(r_nx), r_nx, dim=3))
        Proj_s = torch.einsum('npij,npmj->npmi', Proj, rel_vel)
        grad_s_va = (-Proj_s / (s_norm.unsqueeze(3) * dt)
                     + omv * torch.cross(
                         n_hat.unsqueeze(2).expand_as(rel_vel),
                         rel_vel / s_norm.unsqueeze(3), dim=3))
        grad_h_over_s = (grad_h / s_norm.unsqueeze(3)
                         - (h_dgu / s_norm.pow(2)).unsqueeze(3)
                           * grad_s_va)
        d_gom_dva = -c_fm.unsqueeze(3) * grad_h_over_s

        Proj_rnx_dt = Proj_rnx / dt
        grad_vb_h = Proj_rnx_dt
        grad_vb_s = (Proj_s / (s_norm.unsqueeze(3) * dt))
        grad_h_over_s_vb = (grad_vb_h / s_norm.unsqueeze(3)
                            - (h_dgu / s_norm.pow(2)).unsqueeze(3)
                              * grad_vb_s)
        d_gom_dvb = -c_fm.unsqueeze(3) * grad_h_over_s_vb
        if gi.numel() > 0:
            d_gom_dvb[:, gi] = 0.0

        dgu_om = torch.einsum('npmc,npmcj->npj', d_gom_dva, J_t_a)
        if li.numel() > 0:
            dgu_om[:, li] = (
                dgu_om[:, li]
                + torch.einsum('npmc,npmcj->npj',
                               d_gom_dvb[:, li], J_t[:, lid_b_lnk]))

        dgu_4 = torch.zeros(N, P, 4, n, device=dev, dtype=dty)
        dgu_4[:, :, :3, :] = dgu_3
        dgu_4[:, :, 3, :] = dgu_om
        J_lift = self._friction_J_lift_batch(t0_dgu, t1_dgu)
        return torch.einsum('npji,npjk->npik', J_lift, dgu_4)

    # ==================================================================
    #  DEBUG — same checks as :class:`simulator.ConvexHullPBADSimulator`
    #          (delegate per env; prints / FD / autograd identical)
    # ==================================================================
    def _reference_single_sim(self):
        """Single-env simulator with parameters aligned to this batch instance."""
        from simulator import ConvexHullPBADSimulator
        sim = ConvexHullPBADSimulator(
            self.robot,
            dt=self.dt,
            barrier_x0=self.x0,
            coef_barrier=self.coef_barrier,
            lm_gamma=self.lm_gamma,
            friction=self.friction,
            gravity=self.gravity,
            max_newton_iter=self.max_iter,
            gtol=self.gtol,
            device=self.device,
            _output=False,
            reject_step_energy_above=self._reject_step_energy_above,
            use_joint_limit_barrier=self._use_joint_limit_barrier,
            barrier_d0=self.barrier_d0,
            implicit=self._implicit,
            use_friction=self._use_friction,
            contact_exclude_tree_distance=self._contact_exclude_tree_distance,
        )
        sim._friction_eps_s = self._friction_eps_s
        sim._friction_eps_n = self._friction_eps_n
        sim._local_verts = self._local_verts.detach().clone()
        return sim

    @staticmethod
    def _debug_wv_row(wv: Optional[torch.Tensor], env_idx: int,
                      batch_n: int) -> Optional[torch.Tensor]:
        if wv is None:
            return None
        if wv.dim() == 4 and wv.shape[0] == batch_n:
            return wv[env_idx].contiguous()
        return wv

    def _debug_num_envs_to_run(
            self, n_envs: Optional[int],
            theta: Optional[torch.Tensor]) -> int:
        if theta is None:
            # Random mode: one C++-style trial per call unless ``n_envs`` set.
            if n_envs is None:
                return 1
            return max(0, min(int(n_envs), self.N))
        cap = self.N if n_envs is None else max(0, min(int(n_envs), self.N))
        if theta.dim() == 1:
            return 1
        if theta.dim() != 2:
            raise ValueError(
                "debug: theta must be [n] or [N,n], got shape "
                f"{tuple(theta.shape)}")
        return max(0, min(cap, theta.shape[0]))

    def debug_energy(
            self,
            scale: float = 0.1,
            custom_delta: Optional[float] = None,
            theta: Optional[torch.Tensor] = None,
            theta_t: Optional[torch.Tensor] = None,
            theta_tm1: Optional[torch.Tensor] = None,
            pd_target: Optional[torch.Tensor] = None,
            kp: float = 100.0,
            kd: float = 10.0,
            max_random_trials: int = 200,
            manifolds: Optional[List[Any]] = None,
            wv_curr_cache: Optional[torch.Tensor] = None,
            wv_last_cache: Optional[torch.Tensor] = None,
            compare_autograd: bool = True,
            normalize_dx: bool = True,
            n_envs: Optional[int] = None) -> None:
        """
        Same behaviour as :meth:`simulator.ConvexHullPBADSimulator.debug_energy`
        (C++ ``debugEnergy`` / ``DEBUG_GRADIENT`` lines, DE / DDE / DDE-L / …).

        Runs that check independently for ``n_env`` rows (default: all ``self.N``,
        capped by batch size).  States may be:

        * ``None`` for all four of ``theta, theta_t, theta_tm1, pd_target`` —
          random sampling per env (same as single-env).
        * 1D tensors ``[n]`` — one shared pose; a **single** run (``n_envs``
          ignored except must be ≥ 1).
        * 2D tensors ``[N_row, n]`` — row ``e`` is env ``e``; number of runs is
          ``min(n_envs, N_row, self.N)``.

        World vertices caches: optional ``[N, L, M, 3]`` (batch FK), sliced per
        env; or ``[L, M, 3]`` shared across runs.

        ``manifolds`` is passed through unchanged (single-env list); typically
        ``None`` when checking batched random poses.
        """
        n_run = self._debug_num_envs_to_run(n_envs, theta)
        if n_run == 0:
            print("BatchSimulator.debug_energy: n_envs=0, nothing to do.")
            return

        vec_mode = (
            theta is not None and theta.dim() == 1)
        for e in range(n_run):
            print(
                f"\n{'=' * 70}\n"
                f"  BatchSimulator.debug_energy  env {e} / {n_run - 1}  "
                f"(batch_N={self.N})\n"
                f"{'=' * 70}")
            ref = self._reference_single_sim()
            th = theta
            tht = theta_t
            thtm = theta_tm1
            pd_ = pd_target
            if not vec_mode and theta is not None:
                th = theta[e]
                tht = theta_t[e]
                thtm = theta_tm1[e]
                pd_ = pd_target[e]
            wvc = self._debug_wv_row(wv_curr_cache, e, self.N)
            wvl = self._debug_wv_row(wv_last_cache, e, self.N)
            ref.debug_energy(
                scale=scale,
                custom_delta=custom_delta,
                theta=th,
                theta_t=tht,
                theta_tm1=thtm,
                pd_target=pd_,
                kp=kp,
                kd=kd,
                max_random_trials=max_random_trials,
                manifolds=manifolds,
                wv_curr_cache=wvc,
                wv_last_cache=wvl,
                compare_autograd=compare_autograd,
                normalize_dx=normalize_dx,
            )
        print(
            f"\nBatchSimulator.debug_energy complete  "
            f"({n_run} env(s), batch_N={self.N}).")

    def debug_backward(
            self,
            scale: float = 0.1,
            custom_delta: Optional[float] = None,
            theta_t: Optional[torch.Tensor] = None,
            theta_tm1: Optional[torch.Tensor] = None,
            pd_target: Optional[torch.Tensor] = None,
            kp: float = 100.0,
            kd: float = 10.0,
            max_random_trials: int = 200,
            manifolds: Optional[List[Any]] = None,
            compare_autograd_backward: bool = True,
            normalize_dx: bool = True,
            n_envs: Optional[int] = None) -> None:
        """
        Same behaviour as
        :meth:`simulator.ConvexHullPBADSimulator.debug_backward`
        (DTDL / DTDLL / DTDP / DTDXL, ``DEBUG_GRADIENT`` + optional autograd).

        Tensor layout matches :meth:`debug_energy` (``None`` / 1D / 2D per row).
        """
        n_run = self._debug_num_envs_to_run(n_envs, theta_t)
        if n_run == 0:
            print("BatchSimulator.debug_backward: n_envs=0, nothing to do.")
            return

        vec_mode = (
            theta_t is not None and theta_t.dim() == 1)
        for e in range(n_run):
            print(
                f"\n{'=' * 70}\n"
                f"  BatchSimulator.debug_backward  env {e} / {n_run - 1}  "
                f"(batch_N={self.N})\n"
                f"{'=' * 70}")
            ref = self._reference_single_sim()
            tht = theta_t
            thtm = theta_tm1
            pd_ = pd_target
            if not vec_mode and theta_t is not None:
                tht = theta_t[e]
                thtm = theta_tm1[e]
                pd_ = pd_target[e]
            ref.debug_backward(
                scale=scale,
                custom_delta=custom_delta,
                theta_t=tht,
                theta_tm1=thtm,
                pd_target=pd_,
                kp=kp,
                kd=kd,
                max_random_trials=max_random_trials,
                manifolds=manifolds,
                compare_autograd_backward=compare_autograd_backward,
                normalize_dx=normalize_dx,
            )
        print(
            f"\nBatchSimulator.debug_backward complete  "
            f"({n_run} env(s), batch_N={self.N}).")

    def debug_verify_theta_derivatives_fd(
            self,
            theta: torch.Tensor,
            theta_t: torch.Tensor,
            theta_tm1: torch.Tensor,
            pd_target: torch.Tensor,
            kp: float = 100.0,
            kd: float = 10.0,
            manifolds: Optional[List[Any]] = None,
            eps: Optional[float] = None,
            eps_hess: Optional[float] = None,
            wv_curr_cache: Optional[torch.Tensor] = None,
            wv_last_cache: Optional[torch.Tensor] = None,
            max_n_full_hess: int = 24,
            n_hess_random: int = 8,
            compare_hvp_autograd: bool = True,
            compare_cross_hessian: bool = True,
            compare_cross_autograd: bool = True,
            verbose: bool = True,
            n_envs: Optional[int] = None) -> List[Dict[str, float]]:
        """
        Same checks as
        :meth:`simulator.ConvexHullPBADSimulator.debug_verify_theta_derivatives_fd`,
        one report per environment row (optional ``n_envs`` cap).
        """
        n_run = self._debug_num_envs_to_run(n_envs, theta)
        if n_run == 0:
            print("BatchSimulator.debug_verify_theta_derivatives_fd: n_envs=0.")
            return []

        vec_mode = theta.dim() == 1
        outs: List[Dict[str, float]] = []
        for e in range(n_run):
            print(
                f"\n======== env {e} / {n_run - 1}  (batch_N={self.N}) ========")
            ref = self._reference_single_sim()
            th = theta if vec_mode else theta[e]
            tht = theta_t if vec_mode else theta_t[e]
            thtm = theta_tm1 if vec_mode else theta_tm1[e]
            pd_ = pd_target if vec_mode else pd_target[e]
            wvc = self._debug_wv_row(wv_curr_cache, e, self.N)
            wvl = self._debug_wv_row(wv_last_cache, e, self.N)
            outs.append(ref.debug_verify_theta_derivatives_fd(
                th, tht, thtm, pd_,
                kp=kp, kd=kd,
                manifolds=manifolds,
                eps=eps,
                eps_hess=eps_hess,
                wv_curr_cache=wvc,
                wv_last_cache=wvl,
                max_n_full_hess=max_n_full_hess,
                n_hess_random=n_hess_random,
                compare_hvp_autograd=compare_hvp_autograd,
                compare_cross_hessian=compare_cross_hessian,
                compare_cross_autograd=compare_cross_autograd,
                verbose=verbose,
            ))
        print(
            f"\nBatchSimulator.debug_verify_theta_derivatives_fd complete  "
            f"({n_run} env(s)).")
        return outs

    def _print_multi_step_lm_progress(
            self, step_idx: torch.Tensor, outer_it: int) -> None:
        """Per-outer-iter PD substep indices (all envs) when ``_output``."""
        if not self._output:
            return
        vals = step_idx.detach().cpu().tolist()
        print(
            f"  [multi_step outer_iter={outer_it}] PD_substep_idx 【"
            + ",".join(str(int(x)) for x in vals) + "】",
            flush=True)

    # ==================================================================
    #  multi_step_batch — one batched LM trial / iter, async PD substep
    #
    #  Each outer iteration ``it``:
    #    1. Per-env convergence → envs that finished the current PD substep
    #       advance ``(θ_{t-1}, θ_t, pd)`` to the next row of ``pd_seq``
    #       immediately (others keep solving the same substep).
    #    2. One **batched** Newton direction + trial + accept/reject over all
    #       ``N`` envs (GPU parallelism); envs with no active substep get
    #       ``live=False`` and zero delta for that iter.
    #    3. Recompute E, g, H if any env advanced or accepted.
    #
    #  If every env just advanced (no ``live`` envs this iter), skip the
    #  useless trial and only refresh E, g, H (async timestep roll-forward).
    # ==================================================================
    @torch.no_grad()
    def multi_step_batch(
            self, theta_t, theta_tm1, pd_seq,
            kp: float = 100.0, kd: float = 10.0,
            on_env_advance: Optional[Callable[
                [torch.Tensor, torch.Tensor, torch.Tensor], None]] = None):
        """Returns ``(traj, theta_final)`` with ``traj`` ``[T+1, N, n]`` and last LM
        iterate ``theta_final`` ``[N, n]`` (caller may chain another call using
        ``theta_final`` / ``traj[-2]`` as the next ``theta_t`` / ``theta_tm1``).

        If ``on_env_advance`` is set, it is invoked whenever at least one
        environment finishes its current PD substep and advances to the next
        (async LM).  Arguments are:

        - ``theta`` ``[N, n]`` — current integrated pose for all envs;
        - ``step_idx`` ``[N]`` long — per-env count of **completed** PD substeps
          in **this** ``pd_seq`` (already incremented);
        - ``advance`` ``[N]`` bool — which envs advanced on this trigger.
        """
        dev, dty = self.device, self.dtype
        N, n = self.N, self.n_dof

        theta_t   = theta_t.to(dtype=dty, device=dev)
        theta_tm1 = theta_tm1.to(dtype=dty, device=dev)
        pd_seq    = pd_seq.to(dtype=dty, device=dev)
        # [T, n] shared targets -> [T, N, n]; [T, N, n] kept (per-env PD trajectories)
        if pd_seq.ndim == 2:
            T = pd_seq.shape[0]
            if pd_seq.shape[1] != n:
                raise ValueError(f"pd_seq expects dof {n}, got {pd_seq.shape[1]}")
            pd_seq = pd_seq.unsqueeze(1).expand(T, N, n).contiguous()
        elif pd_seq.ndim == 3:
            T = pd_seq.shape[0]
            if pd_seq.shape[1] != N or pd_seq.shape[2] != n:
                raise ValueError(
                    f"pd_seq [T,N,n] expects N={N}, n={n}, got {tuple(pd_seq.shape)}")
        else:
            raise ValueError("pd_seq must be [T, n] or [T, N, n]")

        traj     = torch.zeros(T + 1, N, n, device=dev, dtype=dty)
        traj[0]  = theta_t
        step_idx = torch.zeros(N, device=dev, dtype=torch.long)
        cur_t    = theta_t.clone()
        cur_tm1  = theta_tm1.clone()
        theta    = cur_t.clone()
        env_range = torch.arange(N, device=dev)
        cur_pd   = pd_seq[0, env_range, :].clone()
        done     = torch.zeros(N, device=dev, dtype=torch.bool)
        alpha    = self._alpha.clone()
        nu       = torch.full((N,), 2.0, device=dev, dtype=dty)
        it_accepted = torch.zeros(N, device=dev, dtype=torch.long)

        alpha_max, alpha_min = 1e20, 1e-6
        budget    = self.max_iter
        max_total = self.max_iter * 10 * (T + 1)

        # Planes from ``default_theta`` are not valid for per-env ``cur_t``.
        self._reinit_p_warm_from_theta_batch(cur_t)

        # ---- initial batched compute (all N) ----
        wv_curr     = self._get_wv_batch(cur_t)
        wv_last     = self._get_wv_batch(cur_tm1)
        active_mask = self._detect_contacts_batch(theta)
        pn_friction_snap = self._p_warm.detach().clone()
        E, g, H     = self._compute_all_batch(
            theta, active_mask, pn_friction_snap,
            wv_curr, wv_last, cur_pd, cur_t, kp, kd)

        if self._output:
            print(
                f"[multi_step_batch] N={N}  T={T}  (each outer iter: "
                f"【step_idx…】 then per-env LM lines)",
                flush=True)

        self._output_lm_headers(E, g, alpha, active_mask)

        for it in range(max_total):
            # ======== 1. Convergence check & advance (match single-env) ========
            g_max = g.abs().amax(1)
            cvg     = (g_max < self.gtol) & ~done
            exhaust = (it_accepted >= budget) & ~done
            fail    = (alpha >= alpha_max) & ~done
            advance = cvg | exhaust | fail

            if advance.any():
                adv_ids = advance.nonzero(as_tuple=True)[0]
                traj[(step_idx[adv_ids] + 1).long(), adv_ids] = theta[adv_ids]

                step_idx = step_idx + advance.long()
                done = done | (step_idx >= T)
                if on_env_advance is not None and advance.any():
                    on_env_advance(
                        theta.detach().clone(),
                        step_idx.detach().clone(),
                        advance.detach().clone())
                if done.all():
                    # End-of-iter progress (else break skips tail where this runs).
                    self._print_multi_step_lm_progress(step_idx, it)
                    break

                adv    = advance & ~done
                adv_n1 = adv.unsqueeze(1)
                cur_tm1 = torch.where(adv_n1, cur_t, cur_tm1)
                cur_t   = torch.where(adv_n1, theta,  cur_t)
                theta   = torch.where(adv_n1, cur_t,  theta)
                safe_idx = step_idx.clamp(max=T - 1)
                cur_pd = pd_seq[safe_idx, env_range, :]

                nu = torch.where(adv, torch.tensor(
                    2.0, device=dev, dtype=dty), nu)
                it_accepted = torch.where(advance,
                                          torch.zeros_like(it_accepted),
                                          it_accepted)
                if adv.any():
                    pn_friction_snap = pn_friction_snap.clone()
                    pn_friction_snap[adv] = (
                        self._p_warm[adv].detach().clone())

                # Refresh wv_curr/wv_last and E,g,H for the new substep
                wv_curr     = self._get_wv_batch(cur_t)
                wv_last     = self._get_wv_batch(cur_tm1)
                active_mask = self._detect_contacts_batch(theta)
                E, g, H     = self._compute_all_batch(
                    theta, active_mask, pn_friction_snap,
                    wv_curr, wv_last, cur_pd, cur_t, kp, kd)

                if self._output and adv.any():
                    self._output_lm_headers(E, g, alpha, active_mask, adv)

            if done.all():
                self._print_multi_step_lm_progress(step_idx, it)
                break

            # ======== 2. Batched search direction (all N) ========
            live   = ~done & ~advance
            live_f = live.unsqueeze(1).to(dty)

            # All non-finished envs advanced this iter → no trial; skip.
            if not live.any():
                self._print_multi_step_lm_progress(step_idx, it)
                continue

            if self._implicit:
                delta, g_bar, H_bar = self._schur_update_batch(
                    g, H, active_mask, alpha)
            else:
                delta, H_bar = self._solve_theta_direct_batch(g, H, alpha)
            delta = delta * live_f

            # ======== 3. Batched trial step + E_try (all N) ========
            th_try = theta + delta
            p_try  = self._p_warm.clone()
            u_try  = self._u_warm.clone()
            if self._implicit:
                self._back_substitute_batch(
                    p_try, u_try, delta, active_mask, alpha)
            active_try = self._detect_contacts_batch(th_try, p_try)
            E_try = self._eval_energy_batch(
                th_try, p_try, u_try, active_try, pn_friction_snap,
                wv_curr, wv_last, cur_pd, cur_t, kp, kd)
            penetrating = self._penetration_fixed_pu_batch(
                th_try, p_try, active_try)

            # ======== 4. Per-env accept / reject (exact mirror of single-env) ========
            if self._implicit:
                H_pred = H_bar - alpha.unsqueeze(1).unsqueeze(2) * self._eye_n
                pred = (delta * (
                    g_bar + 0.5 * torch.einsum('nij,nj->ni', H_pred, delta))).sum(1)
            else:
                pred = (delta * (
                    g + 0.5 * torch.einsum('nij,nj->ni', H, delta))).sum(1)
            rho = torch.where(pred.abs() > 1e-30,
                              (E_try - E) / pred,
                              torch.zeros_like(pred))
            acc = (torch.isfinite(E_try) & (E_try < E) & (rho > 0) & live
                   & ~penetrating)
            if self._reject_step_energy_above is not None:
                thr = float(self._reject_step_energy_above)
                acc = acc & ~(torch.isfinite(E_try) & (E_try > thr))
            rej  = (~acc) & live

            theta        = torch.where(acc.unsqueeze(1), th_try, theta)
            self._p_warm = torch.where(
                acc[:, None, None], p_try, self._p_warm)
            self._u_warm = torch.where(
                acc[:, None, None], u_try, self._u_warm)
            E = torch.where(acc, E_try, E)

            fac = torch.clamp(
                1.0 - (2.0 * rho - 1.0).pow(3), min=1.0/3.0)
            alpha = torch.where(
                acc, (alpha * fac).clamp(alpha_min, alpha_max), alpha)
            nu = torch.where(
                acc, torch.tensor(2.0, device=dev, dtype=dty), nu)
            alpha = torch.where(rej, alpha * nu, alpha)
            nu = torch.where(rej, nu * 2.0, nu)

            it_accepted = it_accepted + acc.long()

            self._output_lm_reject_fail_all(
                live, rej, acc, E_try, alpha, nu, penetrating, alpha_max)

            # ======== 5. Recompute E,g,H after accept (wv unchanged) ========
            if acc.any():
                active_mask = self._detect_contacts_batch(theta)
                E, g, H     = self._compute_all_batch(
                    theta, active_mask, pn_friction_snap,
                    wv_curr, wv_last, cur_pd, cur_t, kp, kd)
                g_row = g.abs().amax(1)
                self._output_lm_accept_converged_all(
                    acc, it_accepted, E, g, alpha, rho, active_mask, g_row)

            self._print_multi_step_lm_progress(step_idx, it)

        # ---- fill remaining trajectory ----
        for i in range(N):
            s = step_idx[i].item()
            if s < T:
                traj[s + 1:, i] = theta[i]

        self._alpha = alpha
        if self._output:
            print(f"  [multi] done  total_iters={it+1}  "
                  f"final_step_idx={step_idx.detach().cpu().tolist()}",
                  flush=True)
        # Final integrated pose [N, n] after last LM state.
        return traj, theta.clone()
