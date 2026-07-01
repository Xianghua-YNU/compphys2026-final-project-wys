"""
solvers.py - 核心数值算法模块

包含三个递进的 Monte Carlo 模拟器：
1. 一维随机游走 (RandomWalk1D)
2. 二维随机游走 (RandomWalk2D)
3. 格点 SIR 模型 (SIRModel)

所有算法独立于可视化逻辑，便于测试和复用。
"""

import numpy as np
from typing import Tuple, List, Optional


class RandomWalk1D:
    """
    一维随机游走模拟器

    物理模型：
    - 粒子在每一步以等概率 1/2 向左或向右移动步长 a
    - 经过 N 步后，均方位移满足 <x^2> = 2Dt，其中 D = a^2/(2τ)

    验证方案：
    - 系综平均：重复 M 次独立轨迹，计算平均位移和均方位移
    - 与理论高斯分布对比
    """

    def __init__(self, num_steps: int, num_trials: int, step_size: float = 1.0, seed: Optional[int] = None):
        """
        初始化一维随机游走模拟器

        Parameters
        ----------
        num_steps : int
            单次游走的步数 N
        num_trials : int
            独立轨迹数量 M（用于系综平均）
        step_size : float
            每一步的步长 a
        seed : int, optional
            随机种子，用于保证结果可复现
        """
        self.num_steps = num_steps
        self.num_trials = num_trials
        self.step_size = step_size
        self.rng = np.random.default_rng(seed)

        # 存储所有轨迹的位置 (num_trials, num_steps + 1)
        self.trajectories = None
        # 存储均方位移随步数的变化
        self.msd = None
        # 存储最终位移分布
        self.final_positions = None

    def run(self) -> np.ndarray:
        """
        执行一维随机游走模拟

        Returns
        -------
        np.ndarray
            所有轨迹的位置数组，形状为 (num_trials, num_steps + 1)
        """
        # 初始化位置数组：所有粒子从原点出发
        positions = np.zeros((self.num_trials, self.num_steps + 1))

        # 对每一步，生成随机步长：+1 或 -1（等概率）
        for step in range(1, self.num_steps + 1):
            # 生成随机方向：True=向右(+1)，False=向左(-1)
            steps = self.rng.choice([-1, 1], size=self.num_trials)
            # 乘以步长并累加
            positions[:, step] = positions[:, step - 1] + steps * self.step_size

        self.trajectories = positions
        self.final_positions = positions[:, -1]

        # 计算均方位移 (MSD)
        self.msd = np.mean(positions ** 2, axis=0)

        return positions

    def get_msd(self) -> np.ndarray:
        """返回均方位移随步数的变化数组"""
        if self.msd is None:
            raise RuntimeError("请先调用 run() 方法运行模拟")
        return self.msd

    def get_final_positions(self) -> np.ndarray:
        """返回所有轨迹的最终位移"""
        if self.final_positions is None:
            raise RuntimeError("请先调用 run() 方法运行模拟")
        return self.final_positions


class RandomWalk2D:
    """
    二维随机游走模拟器（多粒子并行扩散）

    物理模型：
    - 粒子在二维格点上等概率选择上下左右四个方向移动
    - 粒子云团半径随时间增长：R(t) ∝ √t
    """

    def __init__(self, num_particles: int, num_steps: int, step_size: float = 1.0, seed: Optional[int] = None):
        """
        初始化二维随机游走模拟器

        Parameters
        ----------
        num_particles : int
            同时扩散的粒子数
        num_steps : int
            总步数
        step_size : float
            每一步的步长 a
        seed : int, optional
            随机种子
        """
        self.num_particles = num_particles
        self.num_steps = num_steps
        self.step_size = step_size
        self.rng = np.random.default_rng(seed)

        self.positions = None  # 形状: (num_particles, num_steps + 1, 2)

    def run(self) -> np.ndarray:
        """
        执行二维随机游走模拟

        Returns
        -------
        np.ndarray
            所有粒子的位置数组，形状为 (num_particles, num_steps + 1, 2)
        """
        # 初始化位置：所有粒子从原点出发
        positions = np.zeros((self.num_particles, self.num_steps + 1, 2))

        for step in range(1, self.num_steps + 1):
            # 随机选择方向：0=右, 1=上, 2=左, 3=下
            directions = self.rng.choice([0, 1, 2, 3], size=self.num_particles)

            # 根据方向更新坐标
            # 右: x+1, 左: x-1, 上: y+1, 下: y-1
            dx = np.zeros(self.num_particles)
            dy = np.zeros(self.num_particles)

            dx[directions == 0] = 1.0   # 右
            dx[directions == 2] = -1.0  # 左
            dy[directions == 1] = 1.0   # 上
            dy[directions == 3] = -1.0  # 下

            positions[:, step, 0] = positions[:, step - 1, 0] + dx * self.step_size
            positions[:, step, 1] = positions[:, step - 1, 1] + dy * self.step_size

        self.positions = positions
        return positions

    def get_positions(self) -> np.ndarray:
        """返回所有粒子的位置轨迹"""
        if self.positions is None:
            raise RuntimeError("请先调用 run() 方法运行模拟")
        return self.positions


class SIRModel:
    """
    格点 SIR 传染病模型（事件驱动 Monte Carlo）

    物理模型：
    - L × L 正方形格点，每个格点处于 S（易感）/ I（感染）/ R（恢复）三种状态之一
    - 每一 Monte Carlo 步随机选取一个格点：
        - 若为 I，则以概率 β 感染一个随机选中的最近邻 S
        - 若为 I，则以概率 γ 恢复为 R
    - 总人数 N = S + I + R 严格守恒

    关键观测：
    - 基本再生数 R₀ = β/γ，阈值 R₀ = 1 决定疾病能否爆发
    - 有限尺寸效应：小系统中随机灭绝概率更高
    """

    def __init__(self, grid_size: int, beta: float, gamma: float,
                 initial_infected: int = 1, num_steps: int = 1000, seed: Optional[int] = None):
        """
        初始化 SIR 格点模型

        Parameters
        ----------
        grid_size : int
            格点尺寸 L（总人口 N = L²）
        beta : float
            感染概率（每个 I-S 接触的传染概率）
        gamma : float
            恢复概率（每个 I 个体每步恢复的概率）
        initial_infected : int
            初始感染人数
        num_steps : int
            Monte Carlo 步数
        seed : int, optional
            随机种子
        """
        self.grid_size = grid_size
        self.N = grid_size ** 2
        self.beta = beta
        self.gamma = gamma
        self.initial_infected = initial_infected
        self.num_steps = num_steps
        self.rng = np.random.default_rng(seed)

        # 状态数组：0=S, 1=I, 2=R
        self.state = None
        # 记录每一时刻的 S, I, R 人数
        self.history = None

    def _get_neighbors(self, idx: int) -> List[int]:
        """
        获取格点 idx 的四个最近邻（上下左右，周期性边界）

        Parameters
        ----------
        idx : int
            格点的一维索引

        Returns
        -------
        List[int]
            四个邻居的一维索引列表
        """
        x = idx % self.grid_size
        y = idx // self.grid_size

        # 上下左右邻居（周期性边界）
        neighbors = [
            (x + 1) % self.grid_size + y * self.grid_size,      # 右
            (x - 1) % self.grid_size + y * self.grid_size,      # 左
            x + ((y + 1) % self.grid_size) * self.grid_size,    # 上
            x + ((y - 1) % self.grid_size) * self.grid_size,    # 下
        ]
        return neighbors

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行 SIR 格点模拟

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (S_history, I_history, R_history)，每个数组形状为 (num_steps + 1,)
        """
        # 初始化状态：全部为 S
        self.state = np.zeros(self.N, dtype=int)

        # 随机选择 initial_infected 个个体设为 I
        infected_indices = self.rng.choice(self.N, size=self.initial_infected, replace=False)
        self.state[infected_indices] = 1

        # 初始化历史记录
        s_history = np.zeros(self.num_steps + 1)
        i_history = np.zeros(self.num_steps + 1)
        r_history = np.zeros(self.num_steps + 1)

        s_history[0] = np.sum(self.state == 0)
        i_history[0] = np.sum(self.state == 1)
        r_history[0] = np.sum(self.state == 2)

        for step in range(1, self.num_steps + 1):
            # 随机选择一个格点
            idx = self.rng.integers(0, self.N)

            if self.state[idx] == 1:  # 选中了一个感染者
                # 随机选择一个邻居
                neighbors = self._get_neighbors(idx)
                neighbor_idx = self.rng.choice(neighbors)

                # 尝试感染邻居
                if self.state[neighbor_idx] == 0:  # 邻居是易感者
                    if self.rng.random() < self.beta:
                        self.state[neighbor_idx] = 1

                # 尝试恢复（感染者在同一 Monte Carlo 步内可能恢复）
                if self.rng.random() < self.gamma:
                    self.state[idx] = 2

            # 记录当前时刻的统计数据
            s_history[step] = np.sum(self.state == 0)
            i_history[step] = np.sum(self.state == 1)
            r_history[step] = np.sum(self.state == 2)

            # 检查总人数守恒
            assert s_history[step] + i_history[step] + r_history[step] == self.N, "总人数不守恒！"

        self.history = (s_history, i_history, r_history)
        return self.history

    def get_history(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 S, I, R 的历史记录"""
        if self.history is None:
            raise RuntimeError("请先调用 run() 方法运行模拟")
        return self.history


def sir_threshold_scan(beta_values: np.ndarray, gamma: float, grid_size: int,
                       num_steps: int, initial_infected: int = 5) -> np.ndarray:
    """
    扫描不同的 beta 值，记录最终恢复人数 R(∞)

    用于观测 R₀ = β/γ 阈值附近的相变行为。

    Parameters
    ----------
    beta_values : np.ndarray
        待扫描的 beta 值数组
    gamma : float
        恢复概率（固定）
    grid_size : int
        格点尺寸
    num_steps : int
        每个 beta 的 Monte Carlo 步数
    initial_infected : int
        初始感染人数

    Returns
    -------
    np.ndarray
        每个 beta 对应的最终恢复人数 R(∞)
    """
    final_R = np.zeros(len(beta_values))

    for i, beta in enumerate(beta_values):
        model = SIRModel(
            grid_size=grid_size,
            beta=beta,
            gamma=gamma,
            initial_infected=initial_infected,
            num_steps=num_steps,
            seed=i * 10  # 每个 beta 使用不同种子
        )
        _, _, r_history = model.run()
        final_R[i] = r_history[-1]

    return final_R