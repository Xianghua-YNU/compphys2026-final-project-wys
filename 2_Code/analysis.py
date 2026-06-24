"""
analysis.py - 分析与可视化模块

负责：
1. 处理原始模拟数据
2. 计算物理量（扩散系数 D、MSD 拟合、相变阈值等）
3. 生成论文所需的图表

本模块与 solvers.py 解耦，可独立运行。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple, Optional

# ============ 修复中文显示问题 ============
# 设置 Matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ========================================


def fit_diffusion_coefficient(msd: np.ndarray, step_size: float = 1.0) -> Tuple[float, float]:
    """
    从均方位移 (MSD) 数据拟合扩散系数 D

    理论关系：MSD(t) = 2Dt

    Parameters
    ----------
    msd : np.ndarray
        均方位移随步数的变化数组，msd[t] = <x²>_t
    step_size : float
        每一步的步长 a（用于将步数转换为物理时间）

    Returns
    -------
    Tuple[float, float]
        (D, D_error) 扩散系数及其拟合误差
    """
    # 使用前 80% 的数据进行拟合（避免后期统计涨落过大）
    n_fit = int(len(msd) * 0.8)

    # 自变量：步数 t（从 0 开始）
    t = np.arange(n_fit)

    # 因变量：MSD(t)
    y = msd[:n_fit]

    # 线性拟合: y = slope * t，其中 slope = 2D
    def linear_func(t, slope):
        return slope * t

    popt, pcov = curve_fit(linear_func, t, y)
    slope = popt[0]
    slope_error = np.sqrt(pcov[0, 0])

    # D = slope / 2（因为 MSD = 2Dt）
    D = slope / 2
    D_error = slope_error / 2

    return D, D_error


def plot_random_walk_1d(positions: np.ndarray, msd: np.ndarray, D: float,
                        step_size: float = 1.0, save_path: Optional[str] = None):
    """
    绘制一维随机游走的结果

    包含两个子图：
    1. 前 50 条轨迹的路径图
    2. 均方位移与拟合曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 子图1：绘制前 50 条轨迹
    n_trajectories = min(50, positions.shape[0])
    for i in range(n_trajectories):
        axes[0].plot(positions[i, :], color='blue', alpha=0.3, linewidth=0.5)
    axes[0].set_xlabel('Steps $t$')
    axes[0].set_ylabel('Position $x$')
    axes[0].set_title(f'1D Random Walk Trajectories (first {n_trajectories})')
    axes[0].grid(True, alpha=0.3)

    # 子图2：均方位移与拟合
    t = np.arange(len(msd))
    axes[1].plot(t, msd, 'b-', label='Simulated MSD', linewidth=2)
    axes[1].plot(t, 2 * D * t, 'r--', label=f'$\\langle x^2 \\rangle = 2Dt$\n$D = {D:.4f}$')
    axes[1].set_xlabel('Steps $t$')
    axes[1].set_ylabel('Mean Square Displacement $\\langle x^2 \\rangle$')
    axes[1].set_title('MSD with Theoretical Fit')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    return fig


def plot_random_walk_2d(positions: np.ndarray, save_path: Optional[str] = None):
    """
    绘制二维随机游走的粒子云团演化

    选取三个时刻：初始、中间、最终
    """
    num_particles = positions.shape[0]
    num_steps = positions.shape[1]

    # 选取四个时刻
    times = [0, num_steps // 4, num_steps // 2, num_steps - 1]
    times_labels = [f't = {t}' for t in times]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for ax, t, label in zip(axes, times, times_labels):
        x = positions[:, t, 0]
        y = positions[:, t, 1]

        ax.scatter(x, y, s=10, alpha=0.5, c='blue')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title(label)
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle('2D Random Walk: Particle Cloud Diffusion')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    return fig


def plot_sir_curve(s_history: np.ndarray, i_history: np.ndarray, r_history: np.ndarray,
                   beta: float, gamma: float, save_path: Optional[str] = None):
    """
    绘制 SIR 模型的时间演化曲线
    """
    t = np.arange(len(s_history))
    N = s_history[0] + i_history[0] + r_history[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t, s_history, 'b-', label='Susceptible S', linewidth=2)
    ax.plot(t, i_history, 'r-', label='Infected I', linewidth=2)
    ax.plot(t, r_history, 'g-', label='Recovered R', linewidth=2)

    ax.set_xlabel('Monte Carlo Steps')
    ax.set_ylabel('Population')
    ax.set_title(f'SIR Model Simulation ($\\beta = {beta:.2f}, \\gamma = {gamma:.2f}, R_0 = {beta/gamma:.2f}$)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 标注总人数
    ax.axhline(y=N, color='gray', linestyle='--', alpha=0.5, label=f'Total N = {N}')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    return fig


def plot_sir_threshold(beta_values: np.ndarray, final_R: np.ndarray,
                       gamma: float, save_path: Optional[str] = None):
    """
    绘制 SIR 模型相变图：最终恢复人数 R(∞) vs β
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(beta_values, final_R, 'bo-', markersize=8, linewidth=2)

    # 标注阈值
    threshold = gamma
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold $R_0 = 1$ ($\\beta = \\gamma = {gamma:.2f}$)')

    ax.set_xlabel('Infection Probability $\\beta$')
    ax.set_ylabel('Final Recovered Population $R(\\infty)$')
    ax.set_title('SIR Model Phase Transition: $R(\\infty)$ vs $\\beta$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    return fig