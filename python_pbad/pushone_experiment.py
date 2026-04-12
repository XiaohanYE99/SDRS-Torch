"""
Push-one MPC 实验脚本（由原始脚本重构为类，行为与参数保持一致）。
运行前请确保可导入 pyPBAD 与项目 Python 目录下的 RBRS。
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# 与 gstest 等脚本一致：从仓库 Python 目录加载 RBRS
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "Python"))

import pyPBAD  # noqa: E402
from RBRS import PosLayer, RBRSLayer  # noqa: E402


class PushOneExperiment:
    """封装 push-one 任务：仿真器、网格形参、MPC 与主循环。"""

    def __init__(self):
        torch.manual_seed(42)
        np.random.seed(42)

        self.pi = math.pi
        self.device = torch.device("cuda")
        self.load = False
        self.dt = 0.04

        self.sim = pyPBAD.MeshBasedPBDSimulator(self.dt)
        self.sim.setCoefBarrier(3e-8)
        self.sim.setfri(0.6)

        self.rad = 0.1
        self.n = 1
        cubes = [[0.5, 0.5, 0.5] for _ in range(self.n)]
        cubes.append([0.1, 0.1, 0.6])
        self.body = pyPBAD.ArticulatedLoader.createPushTask(cubes, False)

        self.floor = pyPBAD.BBoxExact(
            np.array([-30, -30, -200]), np.array([20, 20, -0.26])
        )

        self.nr_d = self.body.nrDOF()
        self.j_limit = self.pi / 8 * 2
        self.kernel = 4
        self.joint_size = 96
        self.half_joint_size = self.joint_size // 2

        self.it_num = 0
        self.horizon = 200

        self.actor = nn.Sequential(
            nn.Linear(40, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Tanh(),
        ).to(self.device)

        if self.load:
            self.actor.load_state_dict(torch.load("controlpush/table_600.pth"))

        self.sim.setArticulatedBody(self.body)
        self.sim.addShape(self.floor)
        self.sim.gravity = np.array([0, 0, -9.81])

        self.mesh = torch.autograd.Variable(
            torch.from_numpy(self.sim.getConvexPoints()).to(self.device), requires_grad=True
        )
        self.shape = torch.autograd.Variable(
            torch.randn(4, dtype=torch.double, device=self.device) * 0.1, requires_grad=True
        )

        self.color = np.empty([self.n * 2 + 3, 3])
        for i in range(self.n * 2 + 3):
            self.color[i] = np.array([0, 1, 0])

        self.render = pyPBAD.SimulatorVisualizer()
        self.render.setLightSize(0)
        self.render.setLightDiffuse(np.array([0.55, 0.55, 0.55]))
        self.render.setArticulateDiffuse(self.color)

        self.lr_control = 3e-3
        self.lr_mesh = 4e-3
        self.lr_shape = 5e-3
        self.opt_control = torch.optim.Adam(
            [{"params": self.actor.parameters(), "lr": self.lr_control, "betas": (0.9, 0.99)}]
        )
        self.loss_fn = torch.nn.MSELoss()
        self.target = torch.autograd.Variable(
            torch.tensor([2.0, 0.0], device=self.device), requires_grad=True
        )

        self.run = RBRSLayer.apply
        self.get_c = PosLayer.apply

        self.j = 1

        self.writer = SummaryWriter("./log_pushone")

        self.buffer_size = 1200
        self.buffer = []

        self.horizon_mpc = 24

        self.traj: list = []

        self._init_pos = np.zeros(12)
        for i in range(self.n):
            self._init_pos[0] = -1
        self._init_pos[7] = 0.2
        self._init_pos[9] = self.pi / 2

        # 由 assemble 更新，供 mpc 使用
        self._trans_mesh = self.assemble(self.mesh, self.shape)

    def assemble(self, mesh, shape):
        _shape = 1 + 0.0 * torch.tanh(shape)
        trans_mesh = 1.0 * mesh
        return trans_mesh

    def mpc(self, a, p, last_p, horizon, iteration, num, init):
        sim = self.sim
        dt = self.dt
        nr_d = self.nr_d
        n = self.n
        trans_mesh = self._trans_mesh
        target = self.target
        device = self.device
        loss_fn = self.loss_fn
        run = self.run

        action = np.zeros_like(a)
        action_rand = np.zeros_like(a)
        action[: horizon - 1, :] = a[1:horizon, :] * 1
        action_rand[: horizon - 1, :] = a[1:horizon, :] * 1

        if num > 0:
            pos = p * 1.0
            last_pos = last_p * 1.0
            sim.pos = pos
            sim.vel = (pos - last_pos) / dt
            for h in range(horizon - 4):
                p_vec = np.zeros(nr_d)
                d_vec = np.zeros(nr_d)
                ct = 10.0 * np.tanh(action_rand[h])
                d_vec[-6:] = np.tanh((pos[-6:] - last_pos[-6:]) / dt + ct * dt) * 1.0
                d_vec[-1] *= 0
                d_vec[-3] *= 0
                p_vec = pos + d_vec * dt
                sim.pos = pos
                sim.vel = (pos - last_pos) / dt
                sim.setPD(p_vec, d_vec, n * 2 + 1, 0, 1e3, 1e1)
                sim.step()
                new_pos = sim.pos
                last_pos = pos
                pos = new_pos
            _pos = pos * 1.0
            _last_pos = last_pos * 1.0

        for n_ in range(num):
            pos = _pos * 1.0
            last_pos = _last_pos * 1.0
            loss = 0
            a = action_rand * 1
            if n_ > 0:
                action_rand[horizon - 4 :] = a[horizon - 4 :] + np.random.rand(4, 6) * 0.8 - 0.4
            for h in range(horizon - 4, horizon):
                p_vec = np.zeros(nr_d)
                d_vec = np.zeros(nr_d)
                ct = 10.0 * action_rand[h]
                d_vec[-6:] = np.tanh((pos[-6:] - last_pos[-6:]) / dt + ct * dt) * 1.0
                p_vec = pos + d_vec * dt
                sim.pos = pos
                sim.vel = (pos - last_pos) / dt
                sim.setPD(p_vec, d_vec, n * 2 + 1, 0, 1e3, 1e1)
                sim.step()
                new_pos = sim.pos
                last_pos = pos
                pos = new_pos
                p0 = torch.from_numpy(pos[0:2]).to(device)
                loss += torch.relu(loss_fn(p0, target[0:2].double()) - 0.5) * 1e4
            if loss < minloss:
                minloss = loss.data
                action = action_rand * 1.0

        lr = 1e-2
        coef = 1e-6
        results = action * 1.0
        for batch in range(len(iteration)):
            minloss = 1e20
            print(action)
            sim.setCoef(coef)
            action = torch.autograd.Variable(torch.from_numpy(results).to(device), requires_grad=True)
            action.requires_grad = True
            opt = torch.optim.Adam([{"params": action, "lr": lr, "betas": (0.3, 0.5)}])
            for it in range(iteration[batch]):
                loss = 0
                loss1 = 0
                pos = torch.from_numpy(p).to(device)
                last_pos = torch.from_numpy(last_p).to(device)
                sim.pos = pos.detach().cpu().numpy()
                sim.vel = (pos.detach().cpu().numpy() - last_pos.detach().cpu().numpy()) / dt
                p_t = torch.from_numpy(p).to(device)
                d_t = torch.zeros(nr_d, device=device)
                last_d = d_t
                for h in range(horizon):
                    ct = 40.0 * torch.tanh(action[h])
                    d_t[-6:] = torch.tanh((pos[-6:] - last_pos[-6:]) / dt / 2 + ct * dt) * 2
                    d_t[-4] *= 0.1 * 0
                    p_t = p_t + d_t * dt
                    new_pos = run(sim, "pushone", p_t, d_t, trans_mesh, pos, last_pos)
                    last_pos = pos
                    pos = new_pos
                    sim.pos = pos.detach().cpu().numpy()
                    sim.vel = (pos.detach().cpu().numpy() - last_pos.detach().cpu().numpy()) / dt
                    vel = (pos - last_pos) / dt
                    last_d = d_t
                p0 = pos[0:2].to(device)
                loss += torch.relu(loss_fn(p0, target[0:2].double()) - 0.01) * 1e4
                loss *= 1e12
                opt.zero_grad()
                loss.backward(retain_graph=True)
                print("it= ", it, " loss= ", loss / 1e12)
                print(torch.norm(action.grad[horizon - 1]), torch.norm(action.grad[0]))
                if loss < minloss:
                    minloss = loss.data
                    results = action.detach().cpu().numpy() * 1.0
                if loss.data < 1e-1:
                    break
                opt.step()
            lr *= 1
            for para in opt.param_groups:
                para["lr"] = lr
            coef *= 1
        return results

    def run_training_loop(self):
        """对应原脚本中 `for it in range(it_num):` 及内层 horizon 循环。"""
        sim = self.sim
        dt = self.dt
        n = self.n
        nr_d = self.nr_d
        mesh = self.mesh
        shape = self.shape
        target = self.target
        device = self.device
        loss_fn = self.loss_fn
        horizon_mpc = self.horizon_mpc
        horizon = self.horizon
        it_num = self.it_num
        init_pos = self._init_pos

        self.actor.train()
        for it in range(it_num):
            if it in [300, 600]:
                self.lr_control *= 0.3
                for para in self.opt_control.param_groups:
                    para["lr"] = self.lr_control

            self.traj = []
            control = []
            sim.resetWithPos(init_pos)
            pos = sim.pos
            last_pos = sim.pos
            sim.vel = (pos - last_pos) / dt
            shape.requires_grad = True
            omega = 0
            trans_mesh = self.assemble(mesh, shape)
            self._trans_mesh = trans_mesh
            last_mesh = 1.0 * trans_mesh
            self.traj.append(pos)

            last_a = (np.random.rand(horizon_mpc, 6) * 0.0 - 0.5) * 0.01
            last_a[:, 0] = -0.05
            p = pos * 1.0
            d = np.zeros(nr_d)
            last_d = d

            for i in range(horizon):
                sim.setCoef(1e-6)
                loss = 0
                print("#########################: ", i)

                sub_it = [50]
                if i % 1 == 0:
                    action = self.mpc(last_a, pos, last_pos, horizon_mpc, sub_it, 0, True)
                last_a = action * 1.0

                ct = 40.0 * np.tanh(action[i % 1])
                d[-6:] = np.tanh((pos[-6:] - last_pos[-6:]) / dt / 2 + ct * dt) * 2
                d[-4] *= 0.1 * 0
                p = p + d * dt
                last_d = d
                sim.pos = pos
                sim.vel = (pos - last_pos) / dt
                sim.setPD(p, d, n * 2 + 1, 0, 1e3, 1e1)
                sim.step()
                new_pos = sim.pos
                last_pos = pos
                pos = new_pos
                sim.pos = pos
                sim.vel = (pos - last_pos) / dt
                if i % 1 == 0:
                    self.traj.append(new_pos)
                control.append(action[0])
                p0 = torch.from_numpy(pos[0:2]).to(device)
                loss += torch.relu(loss_fn(p0, target[0:2].double()) - 0.01) * 1e4
                print("it= ", i, " loss= ", loss)

    def run_visualize(self):
        """使用实例上的 `self.traj` 渲染；需先训练或自行赋值 `self.traj`。"""
        for i in range(len(self.traj)):
            pos = self.traj[i]
        frame = 0
        self.render.visualize(self.sim, self.traj, 0, 1)


def main():
    exp = PushOneExperiment()
    exp.run_training_loop()
    exp.run_visualize()


if __name__ == "__main__":
    main()
