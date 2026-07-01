"""
main.py - 程序入口

按顺序运行三个模块的模拟，并调用 analysis.py 生成图表。
所有结果保存至 ../1_论文/assets/ 目录下。
"""

import os
import numpy as np
from config import RANDOM_WALK_1D, RANDOM_WALK_2D, SIR_MODEL, SIR_SCAN
from solvers import RandomWalk1D, RandomWalk2D, SIRModel, sir_threshold_scan
from analysis import (
    fit_diffusion_coefficient,
    plot_random_walk_1d,
    plot_random_walk_2d,
    plot_sir_curve,
    plot_sir_threshold,
)


# 创建保存图片的目录（论文 assets 文件夹）
ASSETS_DIR = "../1_论文/assets"
os.makedirs(ASSETS_DIR, exist_ok=True)


def main():
    """运行所有模拟并生成图表"""
    print("=" * 60)
    print("计算物理期末项目：随机游走、扩散与病毒传播的 Monte Carlo 模拟")
    print("=" * 60)

    # ==================== 模块1：一维随机游走 ====================
    print("\n[模块1] 运行一维随机游走模拟...")

    rw1d = RandomWalk1D(
        num_steps=RANDOM_WALK_1D["num_steps"],
        num_trials=RANDOM_WALK_1D["num_trials"],
        step_size=RANDOM_WALK_1D["step_size"],
        seed=RANDOM_WALK_1D["seed"],
    )
    trajectories = rw1d.run()
    msd = rw1d.get_msd()

    # 拟合扩散系数
    D, D_error = fit_diffusion_coefficient(msd, step_size=RANDOM_WALK_1D["step_size"])
    print(f"  扩散系数 D = {D:.6f} ± {D_error:.6f}")

    # 绘图
    plot_random_walk_1d(
        trajectories,
        msd,
        D,
        step_size=RANDOM_WALK_1D["step_size"],
        save_path=os.path.join(ASSETS_DIR, "random_walk_1d.png"),
    )
    print("  图表已保存")

    # ==================== 模块2：二维随机游走 ====================
    print("\n[模块2] 运行二维随机游走模拟...")

    rw2d = RandomWalk2D(
        num_particles=RANDOM_WALK_2D["num_particles"],
        num_steps=RANDOM_WALK_2D["num_steps"],
        step_size=RANDOM_WALK_2D["step_size"],
        seed=RANDOM_WALK_2D["seed"],
    )
    positions_2d = rw2d.run()
    plot_random_walk_2d(
        positions_2d,
        save_path=os.path.join(ASSETS_DIR, "random_walk_2d.png"),
    )
    print("  图表已保存")

    # ==================== 模块3：SIR 格点模型（单个 beta） ====================
    print("\n[模块3] 运行 SIR 格点模型模拟...")

    sir = SIRModel(
        grid_size=SIR_MODEL["grid_size"],
        beta=SIR_MODEL["beta"],
        gamma=SIR_MODEL["gamma"],
        initial_infected=SIR_MODEL["initial_infected"],
        num_steps=SIR_MODEL["num_steps"],
        seed=SIR_MODEL["seed"],
    )
    s_hist, i_hist, r_hist = sir.run()
    final_R = r_hist[-1]
    R0 = SIR_MODEL["beta"] / SIR_MODEL["gamma"]
    print(f"  基本再生数 R₀ = {R0:.2f}")
    print(f"  最终恢复人数 R(∞) = {final_R} / {sir.N}")

    plot_sir_curve(
        s_hist, i_hist, r_hist,
        beta=SIR_MODEL["beta"],
        gamma=SIR_MODEL["gamma"],
        save_path=os.path.join(ASSETS_DIR, "sir_curve.png"),
    )
    print("  图表已保存")

    # ==================== 模块4：SIR 阈值扫描 ====================
    print("\n[模块4] 运行 SIR 阈值扫描...")

    beta_values = np.linspace(SIR_SCAN["beta_min"], SIR_SCAN["beta_max"], SIR_SCAN["beta_num"])
    final_R_scan = sir_threshold_scan(
        beta_values=beta_values,
        gamma=SIR_SCAN["gamma"],
        grid_size=SIR_SCAN["grid_size"],
        num_steps=SIR_SCAN["num_steps"],
        initial_infected=5,
    )

    plot_sir_threshold(
        beta_values,
        final_R_scan,
        gamma=SIR_SCAN["gamma"],
        save_path=os.path.join(ASSETS_DIR, "sir_threshold.png"),
    )
    print("  图表已保存")

    # ==================== 完成 ====================
    print("\n" + "=" * 60)
    print("✅ 所有模拟完成！")
    print(f"📁 图片已保存至: {ASSETS_DIR}")
    print("=" * 60)

    # 打印关键结果总结
    print("\n📊 关键结果摘要:")
    print(f"  1. 一维随机游走扩散系数: D = {D:.6f} ± {D_error:.6f}")
    print(f"  2. SIR 模型 (β={SIR_MODEL['beta']}, γ={SIR_MODEL['gamma']}): R₀ = {R0:.2f}")
    print(f"  3. SIR 阈值扫描完成，共扫描 {len(beta_values)} 个 beta 值")


if __name__ == "__main__":
    main()