#!/usr/bin/env python3
"""
Task 14: RMT unfolded 统计量对比
计算 nearest-neighbor spacing distribution 并与 GUE/Poisson 理论对比
"""

import numpy as np
import matplotlib.pyplot as plt
import mpmath
from pathlib import Path

# 设置输出目录
output_dir = Path(__file__).resolve().parent / "task14_rmt_statistics"
output_dir.mkdir(exist_ok=True)

print("="*80)
print("Task 14: 计算 unfolded RMT 统计量")
print("="*80)

# ==================== 1. 获取 Riemann 零点 ====================
print("\n[1] 加载 Riemann 零点...")
mpmath.mp.dps = 15
N_zeros = 100
riemann_zeros = np.array([float(mpmath.zetazero(i).imag) for i in range(1, N_zeros + 1)])
print(f"    加载 {N_zeros} 个零点，范围: {riemann_zeros[0]:.2f} ~ {riemann_zeros[-1]:.2f}")

# ==================== 2. Unfold (去趋势) ====================
print("\n[2] Unfold 零点序列...")
# Riemann 零点的平均密度遵循 Riemann-von Mangoldt 公式
# 这里用简化版：累积计数函数 N(E) ≈ (E/(2π)) log(E/(2π)) - E/(2π)
# Unfolding: 把 gamma_n 映射到均匀间距的坐标系

def riemann_von_mangoldt_count(E):
    """Riemann-von Mangoldt 公式：累积零点计数"""
    if E <= 0:
        return 0
    return (E / (2 * np.pi)) * np.log(E / (2 * np.pi)) - E / (2 * np.pi) + 7/8

# 对每个零点计算其 unfolded 位置
unfolded_positions = np.array([riemann_von_mangoldt_count(z) for z in riemann_zeros])

print(f"    Unfolded 范围: {unfolded_positions[0]:.2f} ~ {unfolded_positions[-1]:.2f}")
print(f"    平均间距: {np.mean(np.diff(unfolded_positions)):.4f} (理论值应接近 1.0)")

# ==================== 3. 计算 nearest-neighbor spacings ====================
print("\n[3] 计算 nearest-neighbor spacings...")
spacings = np.diff(unfolded_positions)
mean_spacing = np.mean(spacings)

# 归一化到平均间距为 1（标准做法）
normalized_spacings = spacings / mean_spacing

print(f"    原始间距统计: 均值={mean_spacing:.4f}, 标准差={np.std(spacings):.4f}")
print(f"    归一化后统计: 均值={np.mean(normalized_spacings):.4f}, 标准差={np.std(normalized_spacings):.4f}")

# ==================== 4. 理论分布 ====================
print("\n[4] 准备理论分布...")

def gue_wigner_surmise(s):
    """GUE 的 Wigner surmise (nearest-neighbor spacing distribution)"""
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)

def poisson_distribution(s):
    """Poisson 过程的间距分布 (完全随机)"""
    return np.exp(-s)

# ==================== 5. 绘图对比 ====================
print("\n[5] 绘制对比图...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：Histogram + 理论曲线
ax = axes[0]
s_range = np.linspace(0, 3, 300)

# 绘制直方图
counts, bins, _ = ax.hist(normalized_spacings, bins=20, density=True,
                          alpha=0.7, color='steelblue', edgecolor='black',
                          label='Riemann zeros (unfolded)')

# 理论曲线
ax.plot(s_range, gue_wigner_surmise(s_range), 'r-', linewidth=2.5,
        label='GUE (Wigner surmise)', zorder=10)
ax.plot(s_range, poisson_distribution(s_range), 'g--', linewidth=2,
        label='Poisson (random)', zorder=9)

ax.set_xlabel('Normalized spacing s', fontsize=12)
ax.set_ylabel('Probability density P(s)', fontsize=12)
ax.set_title('Nearest-Neighbor Spacing Distribution\n(Riemann Zeros vs RMT)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim(0, 3)
ax.set_ylim(0, 1.2)

# 右图：Cumulative distribution
ax = axes[1]
sorted_spacings = np.sort(normalized_spacings)
cumulative_empirical = np.arange(1, len(sorted_spacings) + 1) / len(sorted_spacings)

ax.plot(sorted_spacings, cumulative_empirical, 'o-', markersize=4,
        linewidth=1.5, label='Riemann zeros (empirical CDF)', color='steelblue')

# 理论累积分布
from scipy.integrate import trapezoid
s_theory = np.linspace(0, 3, 300)
cdf_gue = np.array([trapezoid(gue_wigner_surmise(s_theory[:i+1]), s_theory[:i+1])
                    for i in range(len(s_theory))])
cdf_poisson = 1 - np.exp(-s_theory)

ax.plot(s_theory, cdf_gue, 'r-', linewidth=2.5, label='GUE (theory)', zorder=10)
ax.plot(s_theory, cdf_poisson, 'g--', linewidth=2, label='Poisson (theory)', zorder=9)

ax.set_xlabel('Normalized spacing s', fontsize=12)
ax.set_ylabel('Cumulative probability', fontsize=12)
ax.set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim(0, 3)
ax.set_ylim(0, 1)

plt.tight_layout()
output_file = output_dir / "rmt_spacing_distribution.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"    图像已保存: {output_file}")

# ==================== 6. 统计检验 ====================
print("\n[6] 统计检验...")

# Kolmogorov-Smirnov test
from scipy.stats import kstest

# 生成理论 CDF 的插值函数
from scipy.interpolate import interp1d
gue_cdf_interp = interp1d(s_theory, cdf_gue, bounds_error=False, fill_value=(0, 1))

# KS test
ks_stat_gue, ks_pvalue_gue = kstest(normalized_spacings,
                                     lambda x: gue_cdf_interp(x))
ks_stat_poisson, ks_pvalue_poisson = kstest(normalized_spacings,
                                             lambda x: 1 - np.exp(-x))

print(f"    Kolmogorov-Smirnov 检验:")
print(f"      vs GUE:     D = {ks_stat_gue:.4f}, p-value = {ks_pvalue_gue:.4f}")
print(f"      vs Poisson: D = {ks_stat_poisson:.4f}, p-value = {ks_pvalue_poisson:.4f}")

if ks_pvalue_gue > 0.05:
    print(f"      → Riemann 零点与 GUE 一致 (p > 0.05，不能拒绝)")
else:
    print(f"      → Riemann 零点与 GUE 存在显著差异 (p < 0.05)")

# ==================== 7. 保存数值结果 ====================
results = {
    'n_zeros': N_zeros,
    'mean_spacing_raw': float(mean_spacing),
    'std_spacing_normalized': float(np.std(normalized_spacings)),
    'ks_test_gue': {'statistic': float(ks_stat_gue), 'pvalue': float(ks_pvalue_gue)},
    'ks_test_poisson': {'statistic': float(ks_stat_poisson), 'pvalue': float(ks_pvalue_poisson)},
    'spacings_normalized': normalized_spacings.tolist()
}

import json
results_file = output_dir / "rmt_statistics_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[完成] 数值结果已保存: {results_file}")
print("="*80)
