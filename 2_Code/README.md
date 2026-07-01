# 2_Code/ - 源代码运行说明

本项目包含三个递进的 Monte Carlo 模拟模块，用于研究随机游走扩散现象和 SIR 传染病传播。

## 文件结构

| 文件 | 功能 |
|------|------|
| `config.py` | 物理参数集中配置（修改此文件可调整所有模拟参数） |
| `solvers.py` | 核心算法模块（RandomWalk1D, RandomWalk2D, SIRModel） |
| `analysis.py` | 分析与可视化模块（绘图、拟合、数据处理） |
| `main.py` | 程序入口，按顺序运行所有模块并生成图表 |
| `requirements.txt` | Python 环境依赖 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行所有模块

```bash
python main.py
```

### 3. 输出

- 所有图表将自动保存至 `../1_论文/assets/` 目录
- 控制台将输出关键结果摘要

## 模块说明

### 模块1：一维随机游走

验证爱因斯坦扩散关系 $\langle x^2 \rangle = 2Dt$，拟合扩散系数 $D$。

**可调参数** (在 `config.py` 的 `RANDOM_WALK_1D` 中修改)：
- `num_steps`: 单次游走步数
- `num_trials`: 独立轨迹数量（系综平均）
- `step_size`: 步长

### 模块2：二维随机游走

模拟多粒子并行扩散，可视化粒子云团随时间的演化。

**可调参数** (在 `config.py` 的 `RANDOM_WALK_2D` 中修改)：
- `num_particles`: 粒子数
- `num_steps`: 总步数
- `step_size`: 步长

### 模块3：格点 SIR 模型

事件驱动 Monte Carlo 模拟传染病在格点人群中的传播。

**可调参数** (在 `config.py` 的 `SIR_MODEL` 中修改)：
- `grid_size`: 格点尺寸（总人口 $N = L^2$）
- `beta`: 感染概率
- `gamma`: 恢复概率
- `num_steps`: Monte Carlo 步数

### 模块4：SIR 阈值扫描

扫描不同 $\beta$ 值，观测 $R_0 = \beta/\gamma$ 阈值附近的相变行为。

**可调参数** (在 `config.py` 的 `SIR_SCAN` 中修改)：
- `beta_min` / `beta_max`: 扫描范围
- `beta_num`: 扫描点数
- `grid_size`: 格点尺寸
- `num_steps`: 每个 beta 的 Monte Carlo 步数

## 预期输出

运行完成后，`../1_论文/assets/` 目录下将生成以下图片：

1. `random_walk_1d.png` — 一维随机游走轨迹与 MSD 拟合
2. `random_walk_2d.png` — 二维随机游走粒子云团演化
3. `sir_curve.png` — SIR 模型时间演化曲线
4. `sir_threshold.png` — SIR 模型相变图 ($R(\infty)$ vs $\beta$)

## 复现论文结果

若要复现论文中的所有结果，只需运行：

```bash
python main.py
```

所有随机种子已在 `config.py` 中固定，确保结果完全可复现。

## AI 使用声明

本项目使用 AI 辅助编写了部分代码框架，包括：
- 模块化代码结构设计
- 核心算法实现（随机游走、SIR 模型）
- 可视化函数模板


