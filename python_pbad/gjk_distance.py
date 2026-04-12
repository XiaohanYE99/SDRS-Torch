# Contact activation for convex vertex clouds (link hulls).
#
# Default: **axis-aligned bounding box (AABB)** separation distance between the
# min/max corners of each body's world vertices.  A pair is **active** when
# that separation is **≤** ``contact_dist_thresh`` (boxes overlap or are
# within the margin).  Optional: warm-started **plane** margin (legacy).
#
# This is a coarse proxy for true convex-hull contact; it is cheap (batched
# on GPU for ``detect_contacts_active_mask_batch``).

from __future__ import annotations

from typing import Optional

import torch


def contact_manifold_distance_threshold(barrier_x0: float, barrier_d0: float) -> float:
    """
    C++ ``ConvHullPBADSimulator::detectContact`` cutoff:
    ``2 * (_barrier._x0 + _d0) / (1 - _barrier._x0)``.
    """
    bx = float(barrier_x0)
    denom = max(1e-12, 1.0 - bx)
    return 2.0 * (bx + float(barrier_d0)) / denom


def _aabb_separation_dist(lo_a, hi_a, lo_b, hi_b) -> float:
    """Euclidean gap between two AABBs; 0 if overlapping."""
    g2 = 0.0
    for i in range(3):
        g = 0.0
        if hi_a[i] < lo_b[i]:
            g = float(lo_b[i] - hi_a[i])
        elif hi_b[i] < lo_a[i]:
            g = float(lo_a[i] - hi_b[i])
        g2 += g * g
    return g2 ** 0.5


def _tensor_aabb_separation(lo_a, hi_a, lo_b, hi_b) -> torch.Tensor:
    """Axis-aligned box separation [N] from min/max corners [N,3]."""
    acc = torch.zeros(lo_a.shape[0], device=lo_a.device, dtype=lo_a.dtype)
    for i in range(3):
        hai, loa, hib, lob = hi_a[:, i], lo_a[:, i], hi_b[:, i], lo_b[:, i]
        gx = torch.zeros_like(hai)
        gx = torch.where(hai < lob, lob - hai, gx)
        gx = torch.where(hib < loa, loa - hib, gx)
        acc = acc + gx * gx
    return torch.sqrt(acc.clamp(min=0.0))


@torch.no_grad()
def link_link_aabb_contact_active(
        wv_i: torch.Tensor, wv_j: torch.Tensor, n_i: int, n_j: int,
        contact_dist_thresh: float) -> bool:
    """True if AABB separation between the two links is ≤ threshold."""
    va = wv_i[:n_i]
    vb = wv_j[:n_j]
    t = float(contact_dist_thresh)
    lo_a, _ = va.min(dim=0)
    hi_a, _ = va.max(dim=0)
    lo_b, _ = vb.min(dim=0)
    hi_b, _ = vb.max(dim=0)
    return _aabb_separation_dist(lo_a, hi_a, lo_b, hi_b) <= t


@torch.no_grad()
def link_ground_aabb_contact_active(
        wv_i: torch.Tensor, n_i: int, ground_verts: torch.Tensor,
        contact_dist_thresh: float) -> bool:
    """True if AABB separation between link and ground vertices is ≤ threshold."""
    gv = ground_verts.to(device=wv_i.device, dtype=wv_i.dtype)
    va = wv_i[:n_i]
    t = float(contact_dist_thresh)
    lo_a, _ = va.min(dim=0)
    hi_a, _ = va.max(dim=0)
    lo_b, _ = gv.min(dim=0)
    hi_b, _ = gv.max(dim=0)
    return _aabb_separation_dist(lo_a, hi_a, lo_b, hi_b) <= t


# ---------------------------------------------------------------------------
# Optional diagnostics (vertex–vertex min); not used for default activation.
# ---------------------------------------------------------------------------

@torch.no_grad()
def vertex_min_dist_sq_masked(va: torch.Tensor, vb: torch.Tensor,
                              ma: Optional[torch.Tensor],
                              mb: Optional[torch.Tensor]) -> torch.Tensor:
    """Min squared distance between vertex sets; ma, mb bool [Na], [Nb] or None."""
    if ma is not None:
        va = va[ma]
    if mb is not None:
        vb = vb[mb]
    if va.shape[0] == 0 or vb.shape[0] == 0:
        return torch.tensor(1e12, device=va.device, dtype=va.dtype)
    d = torch.cdist(va.unsqueeze(0), vb.unsqueeze(0)).squeeze(0)
    return d.pow(2).min()


@torch.no_grad()
def link_link_plane_margin_below(
        wv_i: torch.Tensor, wv_j: torch.Tensor, n_i: int, n_j: int,
        contact_dist_thresh: float, p_ws: torch.Tensor) -> bool:
    """Warm plane activation: min plane margin on both sides < threshold."""
    dev, dty = wv_i.device, wv_i.dtype
    va = wv_i[:n_i]
    vb = wv_j[:n_j]
    va_h = torch.cat([va, torch.ones(va.shape[0], 1, device=dev, dtype=dty)], 1)
    vb_h = torch.cat([vb, torch.ones(vb.shape[0], 1, device=dev, dtype=dty)], 1)
    d_a = -(va_h @ p_ws)
    d_b = vb_h @ p_ws
    min_d = min(d_a.min().item(), d_b.min().item())
    return min_d < float(contact_dist_thresh)


@torch.no_grad()
def link_ground_plane_margin_below(
        wv_i: torch.Tensor, n_i: int, ground_h: torch.Tensor,
        contact_dist_thresh: float, p_ws: torch.Tensor) -> bool:
    dev, dty = wv_i.device, wv_i.dtype
    va = wv_i[:n_i]
    va_h = torch.cat([va, torch.ones(va.shape[0], 1, device=dev, dtype=dty)], 1)
    d_a = -(va_h @ p_ws)
    d_b = ground_h @ p_ws
    min_d = min(d_a.min().item(), d_b.min().item())
    return min_d < float(contact_dist_thresh)


def _normalize_contact_mode(mode: str) -> str:
    m = str(mode).lower().strip()
    if m in ('vertex_gpu', 'gjk', 'aabb'):
        return 'aabb'
    return m


@torch.no_grad()
def detect_contacts_active_mask_batch(
        wv: torch.Tensor,
        lid_a: torch.Tensor,
        lid_b: torch.Tensor,
        vc: torch.Tensor,
        contact_dist_thresh: float,
        mode: str,
        parent_child: frozenset,
        ground_verts: Optional[torch.Tensor],
        ground_h: Optional[torch.Tensor],
        p_warm: torch.Tensor,
) -> torch.Tensor:
    """
    Batched active mask [N, P].

    ``aabb`` (and legacy aliases ``vertex_gpu``, ``gjk``): GPU AABB separation
    for all envs; active iff separation ≤ ``contact_dist_thresh``.

    ``plane``: per-env warm-plane margin test.

    ``p_warm`` [N,P,4] for plane mode; ``ground_h`` [Mg,4] for plane ground side.
    """
    dev, dty = wv.device, wv.dtype
    N, P = wv.shape[0], lid_a.shape[0]
    active = torch.zeros(N, P, device=dev, dtype=torch.bool)
    tf = float(contact_dist_thresh)
    mkey = _normalize_contact_mode(mode)

    if mkey == 'aabb':
        gv_exp = None
        if ground_verts is not None:
            gv_exp = ground_verts.to(device=dev, dtype=dty).unsqueeze(0).expand(
                N, -1, -1)
        for k in range(P):
            la = int(lid_a[k].item())
            lb = int(lid_b[k].item())
            if lb >= 0 and (min(la, lb), max(la, lb)) in parent_child:
                continue
            ni = int(vc[la].item())
            va = wv[:, la, :ni, :]
            if lb < 0:
                if gv_exp is None:
                    continue
                lo_a = va.min(dim=1).values
                hi_a = va.max(dim=1).values
                lo_b = gv_exp.min(dim=1).values
                hi_b = gv_exp.max(dim=1).values
                aabb = _tensor_aabb_separation(lo_a, hi_a, lo_b, hi_b)
                active[:, k] = aabb <= tf
            else:
                nj = int(vc[lb].item())
                vb = wv[:, lb, :nj, :]
                lo_a = va.min(dim=1).values
                hi_a = va.max(dim=1).values
                lo_b = vb.min(dim=1).values
                hi_b = vb.max(dim=1).values
                aabb = _tensor_aabb_separation(lo_a, hi_a, lo_b, hi_b)
                active[:, k] = aabb <= tf
        return active

    if mkey == 'plane':
        for k in range(P):
            la = int(lid_a[k].item())
            lb = int(lid_b[k].item())
            if lb >= 0 and (min(la, lb), max(la, lb)) in parent_child:
                continue
            ni = int(vc[la].item())
            for n in range(N):
                pk = p_warm[n, k]
                if lb < 0:
                    if ground_h is None:
                        continue
                    if link_ground_plane_margin_below(
                            wv[n, la], ni, ground_h, tf, pk):
                        active[n, k] = True
                else:
                    nj = int(vc[lb].item())
                    if link_link_plane_margin_below(
                            wv[n, la], wv[n, lb], ni, nj, tf, pk):
                        active[n, k] = True
        return active

    raise ValueError(
        f"Unknown contact_distance_mode: {mode!r} (use 'aabb', 'plane', "
        f"or legacy 'vertex_gpu'/'gjk')")


@torch.no_grad()
def link_ground_separation_debug_line(
        links, wv_L: torch.Tensor, vert_counts: torch.Tensor,
        ground_vertices: torch.Tensor, prefix: str = "[contact-detect]",
) -> str:
    """Per-link AABB separation (link verts vs ground verts) for logging."""
    parts = []
    gv = ground_vertices
    lo_b, _ = gv.min(dim=0)
    hi_b, _ = gv.max(dim=0)
    for i, lk in enumerate(links):
        ni = int(vert_counts[i].item())
        va = wv_L[i, :ni]
        lo_a, _ = va.min(dim=0)
        hi_a, _ = va.max(dim=0)
        d_ab = _aabb_separation_dist(lo_a, hi_a, lo_b, hi_b)
        parts.append(f"{lk.name}:aabb_sep={d_ab:.6f}")
    return f"{prefix} link↔ground | " + " | ".join(parts)
