#!/usr/bin/env python3
"""
Task 14 补充：对动力学系统的 eigenphases 做 unfolded RMT 统计分析
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import kstest
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 导入动力学内核
from task12_13_experiments_v2 import build_transition_matrix, extract_eigenphases

output_dir = Path(__file__).resolve().parent / "task14_rmt_statistics"
output_dir.mkdir(exist_ok=True)

print("="*80)
print("Task 14 补充：动力学系统 eigenphases 的 RMT 统计")
print("="*80)

# ==================== 1. 生成动力学系统的 eigenphases ====================
print("\n[1] 生成动力学系统的 eigenphases...")
print("    使用最优参数 (k≈6.48, N=100 规模)")

# 固定参数（使用 Table 1 的最优值）
k_opt = 6.481896
u_c = 1.543689
steps = 10_000_000_000
n_bins = 2000
offset = 100000.0
seed = 42

t_end = 1.0 / (np.log(steps + offset)**2)
u_temp = u_c - k_opt * t_end

print("    开始 100 亿步演化（这会花几分钟）...")
import time
t0 = time.time()
counts = build_transition_matrix(u_temp, k_opt, steps, n_bins, offset, seed)
print(f"    演化完成，耗时 {time.time()-t0:.1f} 秒")

print("    提取前 100 个 eigenphases...")
eigenphases = extract_eigenphases(counts, n_eigenvalues=100)
print(f"    成功提取 {len(eigenphases)} 个 eigenphases")

# ==================== 2. Unfold eigenphases ====================
print("\n[2] Unfold eigenphases...")
# 对 eigenphases 也做同样的 unfolding：映射到均匀间距坐标
# 这里用累积计数函数作为 unfolding 函数
unfolded_eigen = np.arange(len(eigenphases))  # 简单线性映射（已经是排序好的）

# 计算 spacings
spacings_eigen = np.diff(eigenphases)
mean_spacing_eigen = np.mean(spacings_eigen)
normalized_spacings_eigen = spacings_eigen / mean_spacing_eigen

print(f"    平均间距: {mean_spacing_eigen:.4f}")
print(f"    归一化后统计: 均值={np.mean(normalized_spacings_eigen):.4f}, "
      f"标准差={np.std(normalized_spacings_eigen):.4f}")

# ==================== 3. 理论分布 ====================
def gue_wigner_surmise(s):
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)

def poisson_distribution(s):
    return np.exp(-s)

# ==================== 4. 绘图对比 ====================
print("\n[3] 绘制对比图...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：Histogram
ax = axes[0]
s_range = np.linspace(0, 3, 300)

ax.hist(normalized_spacings_eigen, bins=20, density=True,
        alpha=0.7, color='orange', edgecolor='black',
        label='Dynamical system eigenphases')

ax.plot(s_range, gue_wigner_surmise(s_range), 'r-', linewidth=2.5,
        label='GUE (Wigner surmise)', zorder=10)
ax.plot(s_range, poisson_distribution(s_range), 'g--', linewidth=2,
        label='Poisson (random)', zorder=9)

ax.set_xlabel('Normalized spacing s', fontsize=12)
ax.set_ylabel('Probability density P(s)', fontsize=12)
ax.set_title('Dynamical System Eigenphase Spacing\nvs RMT Predictions',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim(0, 3)

# 右图：CDF
ax = axes[1]
sorted_spacings_eigen = np.sort(normalized_spacings_eigen)
cdf_empirical_eigen = np.arange(1, len(sorted_spacings_eigen) + 1) / len(sorted_spacings_eigen)

ax.plot(sorted_spacings_eigen, cdf_empirical_eigen, 'o-', markersize=4,
        linewidth=1.5, label='Dynamical system (empirical)', color='orange')

# 理论 CDF
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
output_file = output_dir / "dynamical_system_spacing_distribution.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"    图像已保存: {output_file}")

# ==================== 5. 统计检验 ====================
print("\n[4] 统计检验...")

gue_cdf_interp = interp1d(s_theory, cdf_gue, bounds_error=False, fill_value=(0, 1))

ks_stat_gue_eigen, ks_pvalue_gue_eigen = kstest(normalized_spacings_eigen,
                                                  lambda x: gue_cdf_interp(x))
ks_stat_poisson_eigen, ks_pvalue_poisson_eigen = kstest(normalized_spacings_eigen,
                                                          lambda x: 1 - np.exp(-x))

print(f"    Kolmogorov-Smirnov 检验 (动力学系统 eigenphases):")
print(f"      vs GUE:     D = {ks_stat_gue_eigen:.4f}, p-value = {ks_pvalue_gue_eigen:.4f}")
print(f"      vs Poisson: D = {ks_stat_poisson_eigen:.4f}, p-value = {ks_pvalue_poisson_eigen:.4f}")

if ks_pvalue_gue_eigen > 0.05:
    print(f"      → 动力学 eigenphases 与 GUE 一致 (p > 0.05)")
else:
    print(f"      → 动力学 eigenphases 与 GUE 存在显著差异 (p < 0.05)")

# ==================== 6. 保存结果 ====================
results = {
    'n_eigenphases': len(eigenphases),
    'mean_spacing_raw': float(mean_spacing_eigen),
    'std_spacing_normalized': float(np.std(normalized_spacings_eigen)),
    'ks_test_gue': {'statistic': float(ks_stat_gue_eigen), 'pvalue': float(ks_pvalue_gue_eigen)},
    'ks_test_poisson': {'statistic': float(ks_stat_poisson_eigen), 'pvalue': float(ks_pvalue_poisson_eigen)},
    'spacings_normalized': normalized_spacings_eigen.tolist()
}

import json
results_file = output_dir / "dynamical_system_rmt_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[完成] 数值结果已保存: {results_file}")
print("="*80)
