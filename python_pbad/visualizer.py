"""PyVista renderer for ConvexHullPBAD. No simulation logic — call update_frame() from your loop."""

import time
import numpy as np
import torch
from typing import Optional

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

try:
    from scipy.spatial import ConvexHull as _ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from robot import Robot

COLORS = {
    'body': '#4A90D9',
    'torso': '#4A90D9',
    'upper_leg': '#E8833A',
    'thigh': '#E8833A',
    'aux_1': '#E8833A',
    'lower_leg': '#5CB85C',
    'shin': '#5CB85C',
    'tip_': '#5CB85C',
    'foot_': '#C9A227',
    'front_left_leg': '#B87333',
    'front_right_leg': '#B87333',
    'back_leg': '#B87333',
    'right_back_leg': '#B87333',
}
GROUND_COLOR = '#D2B48C'


def _link_color(name: str) -> str:
    for k, c in COLORS.items():
        if k in name:
            return c
    return '#888888'


def _fit_camera_y_up(plotter, points_xyz: np.ndarray) -> None:
    """Place camera so the full point set is in view (world Y-up)."""
    if points_xyz is None or points_xyz.size == 0:
        plotter.camera.position = (0.8, 0.8, 1.2)
        plotter.camera.focal_point = (0, 0.25, 0)
        plotter.camera.up = (0, 1, 0)
        return
    P = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    c = P.mean(axis=0)
    r = float(np.linalg.norm(P - c, axis=1).max())
    r = max(r, 0.12)
    # Oblique view: +X, +Y, +Z from centroid; distance scales with hull radius.
    direction = np.array([0.72, 0.38, 0.58], dtype=np.float64)
    direction /= np.linalg.norm(direction) + 1e-12
    dist = r * 2.75
    pos = c + direction * dist
    plotter.camera.position = tuple(float(x) for x in pos)
    plotter.camera.focal_point = tuple(float(x) for x in c)
    plotter.camera.up = (0, 1, 0)
    if hasattr(plotter.camera, 'view_angle'):
        plotter.camera.view_angle = min(50.0, 38.0 + 12.0 * min(r / 0.5, 2.0))


class SpiderVisualizer:
    """Renderer only. Sim loop lives in main.py."""

    def __init__(self, robot: Robot, window_size=(1400, 900)):
        if not HAS_PYVISTA:
            raise ImportError("pyvista not installed")
        self.robot = robot
        self.window_size = window_size
        self._paused = False

    @property
    def paused(self):
        return self._paused

    # ---- plotter helpers ----

    @staticmethod
    def _convex_hull_mesh(verts_np: np.ndarray) -> 'pv.PolyData':
        if HAS_SCIPY:
            hull = _ConvexHull(verts_np)
            faces = np.column_stack(
                [np.full(len(hull.simplices), 3, dtype=np.int64),
                 hull.simplices]).ravel()
            return pv.PolyData(verts_np.copy(), faces=faces)
        return pv.PolyData(verts_np.copy()).delaunay_3d().extract_surface()

    def _ground_hull_mesh(self) -> Optional['pv.PolyData']:
        if self.robot.ground is None:
            return None
        return self._convex_hull_mesh(
            self.robot.ground.vertices.cpu().to(torch.float32).numpy())

    def _build_scene(self, plotter):
        gm = self._ground_hull_mesh()
        if gm is not None:
            plotter.add_mesh(gm, color=GROUND_COLOR, opacity=0.4,
                             smooth_shading=True, show_edges=True,
                             edge_color='#CCCCCC', line_width=0.5)
        else:
            plotter.add_mesh(
                pv.Plane(center=(0, 0, 0), direction=(0, 1, 0),
                         i_size=4.0, j_size=4.0, i_resolution=20, j_resolution=20),
                color=GROUND_COLOR, opacity=0.4, show_edges=True,
                edge_color='#CCCCCC', line_width=0.5)
        link_data = []
        for link in self.robot.links:
            vn = link.local_vertices.cpu().to(torch.float32).numpy()
            mesh = self._convex_hull_mesh(vn)
            base_pts = mesh.points.copy()
            color = _link_color(link.name)
            plotter.add_mesh(mesh, color=color, opacity=0.85, smooth_shading=True,
                             show_edges=(vn.shape[0] <= 12),
                             edge_color='#333333', line_width=0.8)
            link_data.append((mesh, base_pts, color))
        return link_data

    def _update_meshes(self, link_data, theta: torch.Tensor):
        with torch.no_grad():
            transforms = self.robot.forward_kinematics(theta)
        for i, (mesh, base_pts, _) in enumerate(link_data):
            T = transforms[i].cpu().to(torch.float32).numpy()
            mesh.points = (T[:3, :3] @ base_pts.T).T + T[:3, 3]

    def _world_points_for_camera(
            self, link_data, theta: torch.Tensor) -> np.ndarray:
        """All link hull vertices + ground (world frame) for framing."""
        self._update_meshes(link_data, theta)
        parts = [m.points.copy() for m, _, _ in link_data]
        if self.robot.ground is not None:
            gv = self.robot.ground.vertices.cpu().to(torch.float32).numpy()
            parts.append(gv)
        return np.vstack(parts) if parts else np.zeros((0, 3), dtype=np.float64)

    # ---- single-env public API ----

    def open_plotter(self, theta_init: torch.Tensor, title='ConvexHullPBAD'):
        plotter = pv.Plotter(window_size=self.window_size, title=title)
        plotter.set_background('white')
        link_data = self._build_scene(plotter)
        plotter.add_text('...', position='upper_left', font_size=14,
                         color='black', name='status')
        plotter.add_key_event('space',
                              lambda: setattr(self, '_paused', not self._paused))
        _fit_camera_y_up(plotter, self._world_points_for_camera(link_data, theta_init))
        plotter.show(interactive_update=True, auto_close=False)
        return plotter, link_data

    def update_frame(self, plotter, link_data, theta, status: str):
        """Returns False if window was closed."""
        self._update_meshes(link_data, theta)
        plotter.remove_actor('status')
        plotter.add_text(status, position='upper_left', font_size=14,
                         color='black', name='status')
        plotter.update()
        return bool(plotter.window_size)

    def wait_if_paused(self, plotter):
        while self._paused:
            plotter.update()
            time.sleep(0.05)

    # ---- batch public API ----

    def open_plotter_batch(self, N, theta_init, title='Batch', spacing=None):
        import math as _m
        if spacing is None:
            spacing = max(0.6, 0.4 + 0.1 * _m.sqrt(N))
        cols = int(_m.ceil(_m.sqrt(N)))
        rows = int(_m.ceil(N / cols))
        x_off = torch.zeros(N)
        z_off = torch.zeros(N)
        for i in range(N):
            x_off[i] = (i % cols) * spacing
            z_off[i] = (i // cols) * spacing

        plotter = pv.Plotter(window_size=self.window_size,
                             title=f'{title} ({N} envs)')
        plotter.set_background('white')

        gm_t = self._ground_hull_mesh()
        if gm_t is not None:
            for i in range(N):
                gm = gm_t.copy()
                pts = gm.points.copy()
                pts[:, 0] += float(x_off[i])
                pts[:, 2] += float(z_off[i])
                gm.points = pts
                plotter.add_mesh(gm, color=GROUND_COLOR, opacity=0.35,
                                 smooth_shading=True, show_edges=True,
                                 edge_color='#CCCCCC', line_width=0.5)
        else:
            ext = max(4.0, max(cols, rows) * spacing + 2.0)
            plotter.add_mesh(
                pv.Plane(center=(cols * spacing / 2, 0, rows * spacing / 2),
                         direction=(0, 1, 0), i_size=ext, j_size=ext),
                color=GROUND_COLOR, opacity=0.4, show_edges=True,
                edge_color='#CCCCCC', line_width=0.5)

        all_ld = []
        for ei in range(N):
            ld_i = []
            for link in self.robot.links:
                vn = link.local_vertices.cpu().to(torch.float32).numpy()
                mesh = self._convex_hull_mesh(vn)
                bp = mesh.points.copy()
                c = _link_color(link.name)
                plotter.add_mesh(mesh, color=c, opacity=0.85, smooth_shading=True,
                                 show_edges=(vn.shape[0] <= 12),
                                 edge_color='#333333', line_width=0.8)
                ld_i.append((mesh, bp, c))
            all_ld.append(ld_i)

        self._update_meshes_batch(all_ld, theta_init, x_off, z_off)
        plotter.add_text('...', position='upper_left', font_size=14,
                         color='black', name='status')
        plotter.add_key_event('space',
                              lambda: setattr(self, '_paused', not self._paused))
        plotter.reset_camera()
        plotter.camera.up = (0, 1, 0)
        try:
            plotter.camera.zoom(0.92)
        except Exception:
            pass
        plotter.show(interactive_update=True, auto_close=False)
        return plotter, all_ld, x_off, z_off

    def _update_meshes_batch(self, all_ld, theta_batch, x_off, z_off):
        N = theta_batch.shape[0]
        with torch.no_grad():
            transforms = self.robot.forward_kinematics_batch(theta_batch)
        for ei in range(N):
            xo, zo = x_off[ei].item(), z_off[ei].item()
            for li, (mesh, bp, _) in enumerate(all_ld[ei]):
                T = transforms[ei, li].cpu().to(torch.float32).numpy()
                pts = (T[:3, :3] @ bp.T).T + T[:3, 3]
                pts[:, 0] += xo
                pts[:, 2] += zo
                mesh.points = pts

    def update_frame_batch(self, plotter, all_ld, theta_batch,
                           x_off, z_off, status):
        self._update_meshes_batch(all_ld, theta_batch, x_off, z_off)
        plotter.remove_actor('status')
        plotter.add_text(status, position='upper_left', font_size=14,
                         color='black', name='status')
        plotter.update()
        return bool(plotter.window_size)

    def snapshot(self, theta, title='', save_path=None):
        pl = pv.Plotter(window_size=self.window_size, off_screen=save_path is not None)
        pl.set_background('white')
        ld = self._build_scene(pl)
        if title:
            pl.add_text(title, position='upper_left', font_size=14, color='black')
        _fit_camera_y_up(pl, self._world_points_for_camera(ld, theta))
        if save_path:
            pl.screenshot(save_path)
        else:
            try:
                pl.enable_trackball_style()
            except Exception:
                pass
            pl.show()
        pl.close()


def render(
        robot: Robot,
        theta: torch.Tensor,
        *,
        save_path: Optional[str] = None,
        title: str = '',
        window_size=(1400, 900),
) -> None:
    """Off-screen PNG when *save_path* is set; otherwise open an interactive window.

    Uses the same mesh layout as :class:`SpiderVisualizer` (convex hull per link).
    """
    if not HAS_PYVISTA:
        raise ImportError(
            'pyvista is required for visualizer.render(); '
            'install with: pip install pyvista')
    th = theta.detach().to(device=robot.device, dtype=robot.dtype)
    SpiderVisualizer(robot, window_size=window_size).snapshot(
        th, title=title, save_path=save_path)
