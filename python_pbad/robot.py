"""Robot: XML parse, FK. World frame Y-up (e.g. ``ant.xml``); gravity uses vertex index 1."""

import math
import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def rotation_matrix_axis_angle_batch(axis: torch.Tensor,
                                     angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues: axis [3], angle [N] → R [N,3,3]."""
    a = axis / (torch.norm(axis) + 1e-12)
    N = angle.shape[0]
    z = torch.zeros(N, device=angle.device, dtype=angle.dtype)
    K = torch.stack([
        torch.stack([z, -a[2].expand(N), a[1].expand(N)], dim=1),
        torch.stack([a[2].expand(N), z, -a[0].expand(N)], dim=1),
        torch.stack([-a[1].expand(N), a[0].expand(N), z], dim=1),
    ], dim=1)                                              # [N, 3, 3]
    s = torch.sin(angle).unsqueeze(1).unsqueeze(2)
    c = torch.cos(angle).unsqueeze(1).unsqueeze(2)
    I = torch.eye(3, device=angle.device, dtype=angle.dtype).unsqueeze(0)
    return I + s * K + (1 - c) * (K @ K)


def euler_to_rotation_matrix_batch(roll: torch.Tensor,
                                   pitch: torch.Tensor,
                                   yaw: torch.Tensor) -> torch.Tensor:
    """Batched ZYX Euler → R [N, 3, 3]."""
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    R = torch.stack([
        torch.stack([cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr], dim=1),
        torch.stack([sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr], dim=1),
        torch.stack([-sp,   cp*sr,            cp*cr],            dim=1),
    ], dim=1)                                              # [N, 3, 3]
    return R


def make_transform_batch(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Batched 4×4 transform from R [N, 3, 3] and t [N, 3]."""
    N = R.shape[0]
    T = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).expand(N, -1, -1).clone()
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


def rotation_matrix_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    return rotation_matrix_axis_angle_batch(axis, angle.unsqueeze(0)).squeeze(0)


def euler_to_rotation_matrix(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    return euler_to_rotation_matrix_batch(
        roll.unsqueeze(0), pitch.unsqueeze(0), yaw.unsqueeze(0)).squeeze(0)


def make_transform(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return make_transform_batch(R.unsqueeze(0), t.unsqueeze(0)).squeeze(0)


def box_vertices(x_half: float, y_half: float, z_half: float,
                 device='cpu', dtype=torch.float32) -> torch.Tensor:
    """8 corner vertices of an axis-aligned box → [8, 3]."""
    signs = torch.tensor([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1],  [1, -1, 1],  [1, 1, -1],  [1, 1, 1],
    ], device=device, dtype=dtype)
    return signs * torch.tensor([x_half, y_half, z_half], device=device, dtype=dtype)


def box_surface_fibonacci_points(
        x_half: float, y_half: float, z_half: float, n: int,
        center: torch.Tensor,
        device='cpu', dtype=torch.float32) -> torch.Tensor:
    """*n* points on the surface of an axis-aligned box (± half extents).

    Fibonacci directions on the unit sphere, then radial projection onto the
    box boundary (same idea as distributing samples on a cuboid).
    """
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    xh = max(float(x_half), 1e-20)
    yh = max(float(y_half), 1e-20)
    zh = max(float(z_half), 1e-20)
    pts = []
    for i in range(n):
        theta = 2.0 * math.pi * i / golden
        cos_phi = 1.0 - 2.0 * (i + 0.5) / n
        sin_phi = math.sqrt(max(0.0, 1.0 - cos_phi * cos_phi))
        sx = sin_phi * math.cos(theta)
        sy = cos_phi
        sz = sin_phi * math.sin(theta)
        ax = abs(sx) / xh
        ay = abs(sy) / yh
        az = abs(sz) / zh
        t = max(ax, ay, az) + 1e-20
        pts.append([sx / t, sy / t, sz / t])
    out = torch.tensor(pts, device=device, dtype=dtype)
    return out + center.to(device=device, dtype=dtype)


def capsule_vertices(radius: float, half_height: float, n_verts: int = 96,
                     center: Tuple[float, ...] = (0., 0., 0.),
                     device='cpu', dtype=torch.float32) -> torch.Tensor:
    """Capsule along +Y; Fibonacci sphere on surface."""
    r = radius
    cyl_half = max(half_height - r, 0.0)
    golden = (1.0 + math.sqrt(5.0)) / 2.0

    pts = []
    for i in range(n_verts):
        theta = 2.0 * math.pi * i / golden
        cos_phi = 1.0 - 2.0 * (i + 0.5) / n_verts
        sin_phi = math.sqrt(max(0.0, 1.0 - cos_phi * cos_phi))

        x = r * sin_phi * math.cos(theta)
        z = r * sin_phi * math.sin(theta)
        y_cap = r * cos_phi
        y = (cyl_half + y_cap) if cos_phi >= 0.0 else (-cyl_half + y_cap)

        pts.append([x + center[0], y + center[1], z + center[2]])

    return torch.tensor(pts, device=device, dtype=dtype)


def mj_zup_vec_to_engine(x: float, y: float, z: float,
                         device, dtype) -> torch.Tensor:
    """MuJoCo Z-up (x,y,z) position/direction → engine Y-up (x,z,y)."""
    return torch.tensor([x, z, y], device=device, dtype=dtype)


def rotation_align_local_y_to_unit(u: torch.Tensor) -> torch.Tensor:
    """R @ e_y = u, u unit 3-vector [3]."""
    dev, dty = u.device, u.dtype
    y = torch.tensor([0., 1., 0.], device=dev, dtype=dty)
    u = u / (torch.norm(u) + 1e-12)
    c = torch.dot(y, u).clamp(-1.0, 1.0)
    s = torch.linalg.norm(torch.cross(y, u))
    if float(s) < 1e-8:
        if float(c) > 0:
            return torch.eye(3, device=dev, dtype=dty)
        return euler_to_rotation_matrix(
            torch.tensor(math.pi, device=dev, dtype=dty),
            torch.zeros((), device=dev, dtype=dty),
            torch.zeros((), device=dev, dtype=dty))
    axis = torch.cross(y, u) / s
    ang = torch.atan2(s, c)
    return rotation_matrix_axis_angle(axis, ang)


def capsule_vertices_fromto(
        p1: torch.Tensor, p2: torch.Tensor, radius: float, n_verts: int,
        device, dtype) -> torch.Tensor:
    """Capsule between sphere centers p1, p2 (MJ ``fromto`` + ``size``); axis along p2-p1."""
    p1 = p1.to(device=device, dtype=dtype).reshape(3)
    p2 = p2.to(device=device, dtype=dtype).reshape(3)
    dvec = p2 - p1
    L = float(torch.norm(dvec).item())
    r = float(radius)
    if L < 1e-8:
        return capsule_vertices(r, r, n_verts, center=tuple(float(x) for x in p1),
                                device=device, dtype=dtype)
    u = dvec / (torch.norm(dvec) + 1e-12)
    # MJ: segment between centers; total length along axis = L + 2r → half_height = L/2 + r
    hh = 0.5 * L + r
    center = 0.5 * (p1 + p2)
    base = capsule_vertices(r, hh, n_verts, center=(0., 0., 0.),
                            device=device, dtype=dtype)
    R = rotation_align_local_y_to_unit(u)
    return (R @ base.T).T + center


@dataclass
class LinkInfo:
    name: str
    mass: float
    local_vertices: torch.Tensor   # [M, 3]
    parent_joint_idx: int = -1
    dof_offset: int = 0
    n_dof: int = 0
    # XML geometry (engine frame) for contact-only resampling — same as main.py
    # ``Robot(xml)`` hull, fewer surface points via ``sample_link_contact_surface_points``.
    geom_kind: str = ''            # '', 'capsule_y', 'capsule_fromto', 'box'
    geom_r: float = 0.0
    geom_hh: float = 0.0           # capsule along +Y (centered)
    geom_center: Optional[torch.Tensor] = None   # [3] offset for box / capsule_y
    geom_p1: Optional[torch.Tensor] = None       # capsule_fromto endpoints
    geom_p2: Optional[torch.Tensor] = None
    geom_box_half: Optional[torch.Tensor] = None  # [3] positive half extents


@dataclass
class JointInfo:
    name: str
    jtype: str
    parent_link: int
    child_link: int
    origin: torch.Tensor
    axis: torch.Tensor
    limit_lower: float = -3.14
    limit_upper: float = 3.14
    dof_offset: int = 0
    n_dof: int = 0


@dataclass
class GroundInfo:
    vertices: torch.Tensor
    friction: float = 0.8


class Robot:
    def __init__(self, xml_path: str, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.links: List[LinkInfo] = []
        self.joints: List[JointInfo] = []
        self.ground: Optional[GroundInfo] = None
        self.n_dof = 0
        self.initial_pos = None
        self.initial_euler: Optional[Tuple[float, float, float]] = None
        self.initial_hinge: Dict[str, float] = {}
        self._parse_xml(xml_path)

    def _parse_xml(self, path: str):
        tree = ET.parse(path)
        root = tree.getroot()
        dev, dt = self.device, self.dtype
        self._frame = root.get('frame', 'engine')

        def conv_vec3(xs: List[float]) -> torch.Tensor:
            if self._frame == 'mj_zup':
                return mj_zup_vec_to_engine(xs[0], xs[1], xs[2], dev, dt)
            return torch.tensor(xs, device=dev, dtype=dt)

        g = root.find('ground')
        if g is not None:
            b = g.find('box')
            xh = float(b.get('x_half'))
            yh = float(b.get('y_half'))
            zh = float(b.get('z_half'))
            ctr = [float(v) for v in b.get('center', '0 0 0').split()]
            verts = box_vertices(xh, yh, zh, dev, dt)
            verts = verts + torch.tensor(ctr, device=dev, dtype=dt)
            self.ground = GroundInfo(vertices=verts, friction=float(g.get('friction', '0.8')))

        link_name_to_idx = {}
        for elem in root.findall('link'):
            name = elem.get('name')
            mass = float(elem.find('mass').text)

            gkind = ''
            gh_r, gh_hh = 0.0, 0.0
            gcenter: Optional[torch.Tensor] = None
            gp1: Optional[torch.Tensor] = None
            gp2: Optional[torch.Tensor] = None
            gboxh: Optional[torch.Tensor] = None

            cap = elem.find('capsule')
            b = elem.find('box')
            if cap is not None:
                r = float(cap.get('radius'))
                nv = int(cap.get('n_verts', '96'))
                if cap.get('fromto') is not None:
                    ft = [float(v) for v in cap.get('fromto').split()]
                    if len(ft) != 6:
                        raise ValueError(f"Link '{name}': capsule fromto needs 6 floats")
                    if self._frame == 'mj_zup':
                        p1 = mj_zup_vec_to_engine(ft[0], ft[1], ft[2], dev, dt)
                        p2 = mj_zup_vec_to_engine(ft[3], ft[4], ft[5], dev, dt)
                    else:
                        p1 = torch.tensor(ft[0:3], device=dev, dtype=dt)
                        p2 = torch.tensor(ft[3:6], device=dev, dtype=dt)
                    verts = capsule_vertices_fromto(p1, p2, r, nv, dev, dt)
                    gkind = 'capsule_fromto'
                    gh_r = r
                    gp1, gp2 = p1.clone(), p2.clone()
                else:
                    hh = float(cap.get('half_height'))
                    ctr = [float(v) for v in cap.get('center', '0 0 0').split()]
                    if self._frame == 'mj_zup':
                        gcenter = mj_zup_vec_to_engine(ctr[0], ctr[1], ctr[2], dev, dt)
                        ctr_e = tuple(float(x) for x in gcenter.tolist())
                    else:
                        gcenter = torch.tensor(ctr, device=dev, dtype=dt)
                        ctr_e = tuple(float(x) for x in ctr)
                    verts = capsule_vertices(r, hh, nv, center=ctr_e, device=dev, dtype=dt)
                    gkind = 'capsule_y'
                    gh_r = r
                    gh_hh = hh
            elif b is not None:
                xh = float(b.get('x_half'))
                yh = float(b.get('y_half'))
                zh = float(b.get('z_half'))
                verts = box_vertices(xh, yh, zh, dev, dt)
                bctr = b.get('center')
                if bctr is not None:
                    off = conv_vec3([float(v) for v in bctr.split()])
                    verts = verts + off
                else:
                    off = torch.zeros(3, device=dev, dtype=dt)
                gkind = 'box'
                gboxh = torch.tensor([xh, yh, zh], device=dev, dtype=dt)
                gcenter = off
            else:
                raise ValueError(f"Link '{name}' has no geometry (box or capsule)")

            link_name_to_idx[name] = len(self.links)
            self.links.append(LinkInfo(
                name=name, mass=mass, local_vertices=verts,
                geom_kind=gkind, geom_r=gh_r, geom_hh=gh_hh,
                geom_center=gcenter, geom_p1=gp1, geom_p2=gp2,
                geom_box_half=gboxh))

        dof_offset = 0
        for elem in root.findall('joint'):
            name = elem.get('name')
            jtype = elem.get('type')
            parent_name = elem.find('parent').text.strip()
            child_name = elem.find('child').text.strip()
            parent_idx = link_name_to_idx.get(parent_name, -1)
            child_idx = link_name_to_idx[child_name]

            origin = torch.zeros(3, device=dev, dtype=dt)
            if elem.find('origin') is not None:
                origin = conv_vec3([float(v) for v in elem.find('origin').text.split()])
            axis = torch.tensor([0, 0, 1], device=dev, dtype=dt)
            if elem.find('axis') is not None:
                axis = conv_vec3([float(v) for v in elem.find('axis').text.split()])
            ll, lu = -3.14, 3.14
            lim = elem.find('limit')
            if lim is not None:
                ll = float(lim.get('lower', '-3.14'))
                lu = float(lim.get('upper', '3.14'))

            if jtype == 'free':
                n_dof = 6
            elif jtype == 'ball':
                n_dof = 3
            elif jtype == 'fixed':
                n_dof = 0
            else:
                n_dof = 1
            jinfo = JointInfo(name=name, jtype=jtype, parent_link=parent_idx, child_link=child_idx,
                              origin=origin, axis=axis, limit_lower=ll, limit_upper=lu,
                              dof_offset=dof_offset, n_dof=n_dof)
            self.joints.append(jinfo)
            self.links[child_idx].parent_joint_idx = len(self.joints) - 1
            self.links[child_idx].dof_offset = dof_offset
            self.links[child_idx].n_dof = n_dof
            dof_offset += n_dof

            if jtype == 'free':
                if elem.find('initial_pos') is not None:
                    ip = [float(v) for v in elem.find('initial_pos').text.split()]
                    if len(ip) != 3:
                        raise ValueError('initial_pos must have 3 values')
                    self.initial_pos = conv_vec3(ip)
                ie = elem.find('initial_euler')
                if ie is not None:
                    self.initial_euler = (
                        float(ie.get('roll', '0')),
                        float(ie.get('pitch', '0')),
                        float(ie.get('yaw', '0')),
                    )

        self.initial_hinge = {}
        for elem in root.findall('initial_hinge'):
            jn = elem.get('joint')
            if not jn:
                raise ValueError('initial_hinge requires joint="..."')
            self.initial_hinge[jn.strip()] = float(elem.get('value', '0'))

        self.n_dof = dof_offset

        if self.ground is not None:
            gv = self.ground.vertices
            lo = gv.min(dim=0).values.detach().cpu()
            hi = gv.max(dim=0).values.detach().cpu()
            ap = os.path.abspath(path)
            print(
                f"[Robot] ground AABB (XML: {ap}): "
                f"min=({lo[0]:.5f}, {lo[1]:.5f}, {lo[2]:.5f}) "
                f"max=({hi[0]:.5f}, {hi[1]:.5f}, {hi[2]:.5f})"
            )

    def forward_kinematics(self, theta: torch.Tensor) -> List[torch.Tensor]:
        dev, dt = theta.device, theta.dtype
        transforms = [torch.eye(4, device=dev, dtype=dt) for _ in range(len(self.links))]
        for joint in self.joints:
            child = joint.child_link
            parent = joint.parent_link
            T_parent = torch.eye(4, device=dev, dtype=dt) if parent < 0 else transforms[parent]
            if joint.jtype == 'free':
                off = joint.dof_offset
                R = euler_to_rotation_matrix(theta[off+3], theta[off+4], theta[off+5])
                T_joint = make_transform(R, theta[off:off+3])
            elif joint.jtype == 'ball':
                off = joint.dof_offset
                R = euler_to_rotation_matrix(theta[off], theta[off+1], theta[off+2])
                T_joint = make_transform(R, joint.origin)
            elif joint.jtype == 'fixed':
                T_joint = make_transform(
                    torch.eye(3, device=dev, dtype=dt), joint.origin)
            else:
                off = joint.dof_offset
                R = rotation_matrix_axis_angle(joint.axis, theta[off])
                T_joint = make_transform(R, joint.origin)
            transforms[child] = T_parent @ T_joint
        return transforms

    def forward_kinematics_batch(self, theta: torch.Tensor) -> torch.Tensor:
        """Batched FK: theta [N, n_dof] -> transforms [N, L, 4, 4]."""
        N = theta.shape[0]
        dev, dt = theta.device, theta.dtype
        L = len(self.links)
        I4 = torch.eye(4, device=dev, dtype=dt).unsqueeze(0).expand(N, -1, -1)
        transforms = [I4.clone() for _ in range(L)]
        for joint in self.joints:
            child = joint.child_link
            parent = joint.parent_link
            T_par = I4 if parent < 0 else transforms[parent]
            off = joint.dof_offset
            if joint.jtype == 'free':
                R = euler_to_rotation_matrix_batch(
                    theta[:, off + 3], theta[:, off + 4], theta[:, off + 5])
                T_jnt = make_transform_batch(R, theta[:, off:off + 3])
            elif joint.jtype == 'ball':
                R = euler_to_rotation_matrix_batch(
                    theta[:, off], theta[:, off + 1], theta[:, off + 2])
                t = joint.origin.unsqueeze(0).expand(N, -1)
                T_jnt = make_transform_batch(R, t)
            elif joint.jtype == 'fixed':
                I3 = torch.eye(3, device=dev, dtype=dt).unsqueeze(0).expand(N, -1, -1)
                t = joint.origin.unsqueeze(0).expand(N, -1)
                T_jnt = make_transform_batch(I3, t)
            else:
                R = rotation_matrix_axis_angle_batch(joint.axis, theta[:, off])
                t = joint.origin.unsqueeze(0).expand(N, -1)
                T_jnt = make_transform_batch(R, t)
            transforms[child] = T_par @ T_jnt
        return torch.stack(transforms, dim=1)              # [N, L, 4, 4]

    def get_local_verts_padded(self) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        sizes = [link.local_vertices.shape[0] for link in self.links]
        max_M = max(sizes)
        L = len(self.links)
        verts = torch.zeros(L, max_M, 3, device=self.device, dtype=self.dtype)
        mask = torch.zeros(L, max_M, device=self.device, dtype=torch.bool)
        for i, link in enumerate(self.links):
            m = sizes[i]
            verts[i, :m] = link.local_vertices
            mask[i, :m] = True
        return verts, mask, sizes

    def get_wv_stacked_batch(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = self.forward_kinematics_batch(theta)           # [N, L, 4, 4]
        verts, mask, _ = self.get_local_verts_padded()     # [L, max_M, 3], [L, max_M]
        R = T[:, :, :3, :3]                                # [N, L, 3, 3]
        t = T[:, :, :3, 3]                                 # [N, L, 3]
        wv = torch.einsum('nlij,lmj->nlmi', R, verts) + t.unsqueeze(2)
        return wv, mask                                    # [N, L, max_M, 3], [L, max_M]

    def default_theta(self) -> torch.Tensor:
        theta = torch.zeros(self.n_dof, device=self.device, dtype=self.dtype)
        if self.initial_pos is not None:
            theta[0:3] = self.initial_pos
        if self.initial_euler is not None:
            r, p, y = self.initial_euler
            theta[3] = r
            theta[4] = p
            theta[5] = y
        for j in self.joints:
            if j.jtype == 'hinge' and j.name in self.initial_hinge:
                theta[j.dof_offset] = self.initial_hinge[j.name]
        return theta

    def random_theta(self, noise_range: float = 0.25) -> torch.Tensor:
        theta = self.default_theta()
        for j in self.joints:
            if j.jtype == 'hinge':
                lo = max(j.limit_lower + 0.1, -noise_range)
                hi = min(j.limit_upper - 0.1, noise_range)
                theta[j.dof_offset] = lo + torch.rand(1, device=self.device, dtype=self.dtype).squeeze() * (hi - lo)
            elif j.jtype == 'ball':
                lo = max(j.limit_lower + 0.1, -noise_range)
                hi = min(j.limit_upper - 0.1, noise_range)
                for e in range(3):
                    theta[j.dof_offset + e] = lo + torch.rand(
                        1, device=self.device, dtype=self.dtype).squeeze() * (hi - lo)
        return theta


def sample_link_contact_surface_points(
        robot: Robot, n_per_link: int = 16,
        device=None, dtype=None) -> Robot:
    """Downsample each link to *n_per_link* vertices **on the XML surface**.

    Uses the same capsule/box definitions as ``Robot._parse_xml`` / ``main.py``
    (Fibonacci-on-surface pills via ``capsule_vertices`` /
    ``capsule_vertices_fromto``; box via ``box_surface_fibonacci_points``),
    not a bounding sphere.  Intended for contact / convex-hull proxies only.
    """
    dev = robot.device if device is None else device
    dt = robot.dtype if dtype is None else dtype
    for link in robot.links:
        k = link.geom_kind
        if k == 'capsule_y':
            c = (link.geom_center if link.geom_center is not None
                 else torch.zeros(3, device=dev, dtype=dt))
            ctr = tuple(float(x) for x in c.tolist())
            v = capsule_vertices(
                link.geom_r, link.geom_hh, n_per_link, center=ctr,
                device=dev, dtype=dt)
        elif k == 'capsule_fromto':
            p1 = link.geom_p1.to(device=dev, dtype=dt)
            p2 = link.geom_p2.to(device=dev, dtype=dt)
            v = capsule_vertices_fromto(
                p1, p2, link.geom_r, n_per_link, dev, dt)
        elif k == 'box':
            h = link.geom_box_half.to(device=dev, dtype=dt)
            c = (link.geom_center if link.geom_center is not None
                 else torch.zeros(3, device=dev, dtype=dt))
            v = box_surface_fibonacci_points(
                float(h[0].item()), float(h[1].item()), float(h[2].item()),
                n_per_link, c.to(device=dev, dtype=dt), dev, dt)
        else:
            v = link.local_vertices
            if v.shape[0] > n_per_link:
                v = v[:n_per_link].clone().to(device=dev, dtype=dt)
            else:
                v = v.to(device=dev, dtype=dt)
        link.local_vertices = v
    return robot
