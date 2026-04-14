# fastSDRS

PyTorch 可微刚体仿真引擎，基于 Convex-Hull Position-Based Articulated Dynamics (ConvexHullPBAD)。

## 依赖

| 包 | 版本 | 用途 |
|---|---|---|
| Python | >= 3.8 | |
| PyTorch | >= 1.12 | 核心计算 |
| NumPy | any | 辅助 |
| PyVista | any | 3D 可视化（可选） |
| SciPy | any | ConvexHull 渲染（可选） |
| TensorBoard | any | `train.py` 训练曲线（可选） |

```bash
pip install torch numpy pyvista scipy tensorboard
```

> GPU 加速请安装对应 CUDA 版本的 PyTorch：https://pytorch.org/get-started/locally/

## 运行

所有命令在 `python_pbad/` 目录下执行：

```bash
cd python_pbad
```

| 命令 | 说明 |
|------|------|
| `python main.py sim [steps] [--render]` | 单环境前向仿真，默认 200 步 |
| `python main.py batch_sim [N] [steps] [--render]` | 一次 `multi_step_batch` 跑满 `steps` 行 PD；`--render` 时 **任一 env 进入下一 PD 子步** 即刷新画面 |
| `python main.py debug [random] [--render]` | 单环境解析梯度/Hessian 有限差分验证 |
| `python main.py batch_debug [N] [random]` | 批量环境逐 env 调用 debug 验证 |
| `python train.py …` | 单环境：IFT 反传优化 PD 参考（正弦参数化或 MLP），见下表；默认机器人 **ant.xml** |

**`train.py` 常用参数**

| 参数 | 说明 |
|------|------|
| `--xml FILE` | 机器人描述（默认 `ant.xml`，与脚本同目录；可换 `spider.xml`） |
| `--pd sin`（默认） | 铰链 PD 由 `PdSineBank`（多谐波 sin）生成 |
| `--pd mlp` | 由 MLP 根据当前 `theta` 前 `D` 维输出铰链加速度，再积分得到 PD 位置目标 |
| `--mlp-in D` | MLP 输入维度，`D=3` 为机身平移 `theta[:3]`，`D=6` 为整块根位姿 |
| `--traj-out PATH` | 训练结束后保存轨迹（默认 `train_traj_final.pt`） |
| `--render-traj PATH` | **仅加载轨迹并回放**，不进行训练 |
| `--logdir DIR` | TensorBoard 日志目录（默认 `runs/train`，相对 `python_pbad/`） |
| `--no-tb` | 关闭 TensorBoard |
| `--text` | 回放时用终端打印关节角，不弹 PyVista |

每个 epoch 开始会调用 `sim.reset_episode_state()`，重置 LM 阻尼与接触 warm-start；训练结束会保存 `theta` 序列及元数据，便于下次 `python train.py --render-traj …` 直接渲染。

示例：

```bash
python main.py sim 300 --render      # 渲染 300 步
python main.py batch_sim 8 200         # 8 环境，multi_step_batch 分块
python main.py debug                 # 梯度正确性验证
python train.py                      # 默认 ant + sin PD，训练并保存轨迹
python train.py --xml spider.xml     # 使用旧版蜘蛛模型
python train.py --pd mlp --mlp-in 6  # MLP + 根 6 维输入
python train.py --render-traj train_traj_final.pt   # 仅回放（优先用轨迹 meta 里的 xml）
tensorboard --logdir runs/train      # 需在 python_pbad 下或写绝对路径
```

## 文件结构

```
python_pbad/
├── ant.xml               默认机器人（MuJoCo 风格四足：躯干 + 4×大腿/小腿）
├── spider.xml            旧版 9-link 蜘蛛（仍可用 ``--xml spider.xml``）
├── robot.py              XML 解析、FK（单/批量）、数据结构
├── simulator.py          单环境 ConvexHullPBAD 仿真器
├── batch_simulator.py    N 环境并行 BatchSimulator
├── visualizer.py         PyVista 3D 渲染（单环境 + 批量网格）
├── main.py               入口（debug / sim，单/批量模式）
├── train.py              单环境 PD 训练：sin / MLP、轨迹保存、TensorBoard、`--render-traj`
└── checkpoints/          （若自行扩展训练脚本可存放检查点）
```

## 算法流程与代码对应

### 总览

每个时间步求解一个关于关节角 θ 的能量最小化问题。接触变量 (p, u) 通过 Schur 消元折叠进 θ 空间，使用 Levenberg-Marquardt 信赖域法迭代求解。收敛后通过隐函数定理 (IFT) 计算反向传播梯度。

### 1. 前向运动学 (FK)

根据关节角 θ 计算各 link 的世界坐标顶点和解析 Jacobian。

| 步骤 | 代码 |
|------|------|
| FK 变换矩阵 | `robot.py` — `Robot.forward_kinematics()` |
| 世界顶点 + FK Jacobian | `simulator.py` — `_compute_fk_jacobian()` |
| 批量 FK | `batch_simulator.py` — `_compute_fk_jacobian_batch()` |

### 2. 接触检测

对每对 link 用 SVM 分离超平面判定活跃接触对，然后用 log-barrier 评估穿透程度。

| 步骤 | 代码 |
|------|------|
| 超平面 SVM 求解 | `simulator.py` — `_solve_separating_plane()` |
| 接触对活跃判定 | `_detect_contacts()` / `_detect_contacts_batch()` |
| Log-barrier 评估 | `barrier_eval()` — 全局函数 |

### 3. 能量、梯度、Hessian 组装

总能量 = 惯性 + 重力 + PD 控制 + 接触 barrier + 摩擦 + 关节限位 barrier。所有梯度和 Hessian 均为手写解析计算（非 autograd）。

| 能量项 | 梯度/Hessian 计算 | 代码 |
|--------|-------------------|------|
| 惯性 E_inertia | 二次型，解析 | `_compute_energy()` |
| 重力 E_gravity | 线性，解析 | `_compute_energy()` |
| PD 控制 E_pd | 二次型，解析 | `_compute_energy()` |
| 接触 barrier E_contact | `g_p`, `H_pp` — barrier 导数 × FK Jacobian | `_compute_energy()` → `barrier_eval()` |
| 摩擦 E_friction | `g_u`, `H_uu`, `H_θu` — Coulomb 平滑近似 | `_compute_manifold_hessians()` |
| 关节限位 | 同 barrier 形式 | `_compute_energy()` |

FK Hessian 修正项（解析二阶导数）：

| 方法 | 代码 |
|------|------|
| Hinge-Hinge | `_compute_HThetaD()` — `a_j × J_k(v)` |
| Euler-Hinge | `_compute_HThetaD()` — `M_r @ J_k(v)` |
| Euler-Euler | `_compute_HThetaD()` — `d²R/(dφ_r dφ_s) @ q_v` |

### 4. Schur 消元 + LM 求解

将 (θ, p, u) 的 KKT 系统通过 Schur 消元压缩为 θ 空间的 n×n 系统，然后用 LM 求解。

| 步骤 | 代码 |
|------|------|
| H_pp / H_uu block 求逆 | `_schur_update()` |
| Schur 消元得 H_θθ_eff, g_θ_eff | `_schur_update()` |
| LM 步 + 信赖域调整 | `step()` 主循环 |
| 接触重检测（每步后） | `_detect_contacts()` |
| p/u 回代 | `_back_substitute()` |

### 5. IFT 反向传播

收敛后对 KKT 稳态条件应用隐函数定理，求 ∂θ\*/∂(θ\_t, θ\_{t-1}, pd\_target)。

| 步骤 | 代码 |
|------|------|
| dg/d(θ\_t, θ\_{t-1}, pd) 组装 | `_backward_normal()` |
| 摩擦交叉项 dg\_u/dθ\_t | `_friction_cross_theta_t()` / `_compute_dgu_dtheta_t()` |
| KKT 线性系统求解 | `_backward_normal()` → `torch.linalg.solve` |
| 梯度输出 | `GradInfo` dataclass |

### 6. 可微训练接口（单环境）

`SimStepFn` 封装了 `torch.autograd.Function`，`forward` 调用 `step()`，`backward` 使用 IFT 梯度，无需构建 autograd 图。批量前向请直接使用 `BatchSimulator.multi_step_batch`（无 `torch.autograd` 封装）。

```python
# 单环境
theta_tp1 = SimStepFn.apply(pd_target, theta_t, theta_tm1, None, sim, kp, kd)
```

### 批量仿真器

`BatchSimulator` 将 N 个环境的张量打包为 `[N, ...]` 形状，在同一 GPU 上并行执行 FK、能量评估、Schur 消元与 LM；主要加速来自数据并行。

## 自定义机器人

```xml
<robot name="my_robot">
  <ground friction="0.8">
    <box x_half="5" y_half="0.25" z_half="5" center="0 -0.25 0"/>
  </ground>
  <link name="body">
    <mass>1.0</mass>
    <box x_half="0.1" y_half="0.05" z_half="0.1"/>
  </link>
  <joint name="root" type="free">
    <parent>world</parent>
    <child>body</child>
    <initial_pos>0 0.5 0</initial_pos>
  </joint>
</robot>
```

支持几何体：`<box>` (8 顶点)、`<capsule>` (Fibonacci 采样表面顶点)。

## License

MIT
