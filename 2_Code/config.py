"""
config.py - 物理参数配置文件

集中管理所有可调物理参数，避免在代码中出现"魔法数字"。
修改此文件即可改变模拟行为，无需改动核心算法代码。
"""

# ============ 模块1：一维随机游走参数 ============
RANDOM_WALK_1D = {
    "num_steps": 500,          # 单次游走的步数
    "num_trials": 1000,        # 独立轨迹数量（用于系综平均）
    "step_size": 1.0,          # 每一步的步长 a
    "seed": 42,                # 随机种子（保证可复现）
}

# ============ 模块2：二维随机游走参数 ============
RANDOM_WALK_2D = {
    "num_particles": 200,      # 同时扩散的粒子数
    "num_steps": 300,          # 总步数
    "step_size": 1.0,          # 每一步的步长 a
    "seed": 123,               # 随机种子
}

# ============ 模块3：SIR 格点模型参数 ============
SIR_MODEL = {
    "grid_size": 50,           # L × L 格点，总人口 N = L^2
    "beta": 0.3,               # 感染概率
    "gamma": 0.1,              # 恢复概率
    "initial_infected": 5,     # 初始感染人数
    "num_steps": 5000,         # Monte Carlo 步数（从 500 增加到 5000）
    "seed": 456,               # 随机种子
}

# ============ SIR 阈值扫描参数 ============
SIR_SCAN = {
    "beta_min": 0.0,           # 扫描的 beta 最小值
    "beta_max": 0.5,           # 扫描的 beta 最大值
    "beta_num": 20,            # 扫描点数
    "gamma": 0.1,              # 固定 gamma
    "grid_size": 50,           # 格点尺寸
    "num_steps": 5000,         # 每个 beta 的 Monte Carlo 步数（从 500 增加到 5000）
}