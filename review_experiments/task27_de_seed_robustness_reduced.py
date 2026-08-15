#!/usr/bin/env python3
"""
Task 27: 缩小规模验证 Strategy-C 的 differential_evolution 非重复性
(回应 58231636 Concern 6 / 58279672 Concern 7 —— Table1 k1=6.481 vs Fig4/5 k1=8.070 的来源)

关键澄清：build_ulam_matrix_anchored 本身是完全确定性的 (初始 x=0.5，无随机数)，
所以矩阵构造这一步不含随机性。差异来源是 differential_evolution 的随机种群初始化 +
'best1bin' 变异策略 + workers=-1 并行评估顺序不确定 —— 也就是优化器本身的随机性，
而不是被优化的目标函数的随机性。

原脚本在 steps=1e10, popsize=256, maxiter=15 下跑一次要 ~630 分钟 (256核)。
这里用大幅缩小的 steps=2e7, popsize=24, maxiter=8，对同一个目标函数跑 5 个
不同的显式 seed，检验:
  (a) 该目标函数在缩小规模下是否仍然是多峰/非凸的 (不同 seed 收敛到不同 k1)
  (b) 这与原论文观察到的 6.481 vs 8.070 两个局部最优是同一类现象，还是
      缩小规模改变了目标函数的形状

注意：缩小规模后的具体 k1 数值不能直接与论文 Table 1 (steps=1e10) 的 6.481/8.070
比较，因为改变 steps 本身会改变目标函数 (matrix 分辨率、有限步数偏差)。这里只
比较"同一个缩小规模目标函数，不同 seed 是否收敛到不同 k1"这一稳健性问题。
"""
import numpy as np
from numba import njit
from scipy.optimize import differential_evolution
import time
import mpmath
import json
from pathlib import Path

output_dir = Path(__file__).resolve().parent / "task27_results"
output_dir.mkdir(exist_ok=True)


def get_exact_riemann_zeros(n_max=100):
    mpmath.mp.dps = 15
    return np.array([float(mpmath.zetazero(i).imag) for i in range(1, n_max + 1)])


@njit(fastmath=True, nogil=True)
def build_ulam_matrix_anchored(u_temp, k_opt, steps, n_bins, offset):
    x = 0.5
    counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    last_bin = int((x + 1.0) / 2.0 * (n_bins - 1))

    warmup_steps = min(200_000, steps // 10)
    for i in range(warmup_steps):
        L_i = np.log(i + offset)
        u_dyn = u_temp + k_opt / (L_i**2)
        x = 1.0 - u_dyn * x**2
        if x > 1.0: x = 0.999
        elif x < -1.0: x = -0.999

    for i in range(warmup_steps, steps + warmup_steps):
        L_i = np.log(i + offset)
        u_dyn = u_temp + k_opt / (L_i**2)
        x = 1.0 - u_dyn * x**2
        if x > 1.0: x = 0.999
        elif x < -1.0: x = -0.999

        current_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
        if 0 <= current_bin < n_bins and 0 <= last_bin < n_bins:
            counts[last_bin, current_bin] += 1
        last_bin = current_bin

    return counts


def extract_phases(counts):
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = counts / row_sums
    vals, _ = np.linalg.eig(P)
    valid_pos = vals[(vals.imag > 1e-5)]
    return np.unwrap(np.sort(np.angle(valid_pos)))


def objective_k_pure_scale(params, target_zeros, u_c, steps, n_bins, offset):
    k_opt = params[0]
    t_end = 1.0 / (np.log(steps + offset) ** 2)
    u_temp = u_c - k_opt * t_end

    counts = build_ulam_matrix_anchored(u_temp, k_opt, steps, n_bins, offset)
    sys_phases = extract_phases(counts)

    N_compare = min(len(target_zeros), len(sys_phases))
    if N_compare < 20:
        return 1e12

    x = sys_phases[:N_compare]
    y = target_zeros[:N_compare]
    slope = np.sum(x * y) / np.sum(x**2)
    predicted = x * slope
    return float(np.mean((predicted - y) ** 2))


if __name__ == "__main__":
    u_c = 1.543689
    scan_steps = 20_000_000       # 缩小 500 倍 (原 1e10)
    scan_offset = 100000.0
    scan_n_bins = 800             # 缩小 2.5 倍 (原 2000)

    true_zeros = get_exact_riemann_zeros(100)

    seeds = [1, 2, 3, 4, 5]
    print(f"{'='*80}")
    print(f"Task 27: 缩小规模 DE 种子稳健性 (steps={scan_steps:.0e}, n_bins={scan_n_bins}, "
          f"popsize=24, maxiter=8, {len(seeds)} 个 seed)")
    print(f"{'='*80}\n")

    results = []
    t_total = time.time()
    for seed in seeds:
        t0 = time.time()
        res = differential_evolution(
            func=objective_k_pure_scale,
            bounds=[(2.0, 15.0)],
            args=(true_zeros, u_c, scan_steps, scan_n_bins, scan_offset),
            strategy="best1bin",
            maxiter=8,
            popsize=24,
            tol=0.001,
            polish=False,
            workers=-1,
            updating="deferred",
            seed=seed,
            disp=False,
        )
        elapsed = time.time() - t0
        best_k = float(res.x[0])
        mse = float(res.fun)
        print(f"[seed={seed}] best_k1={best_k:.4f}  MSE={mse:.4f}  耗时={elapsed:.1f}s")
        results.append({"seed": seed, "best_k1": best_k, "mse": mse, "time_seconds": elapsed})

    k1_values = np.array([r["best_k1"] for r in results])
    summary = {
        "config": {"steps": scan_steps, "n_bins": scan_n_bins, "popsize": 24, "maxiter": 8},
        "results": results,
        "k1_mean": float(np.mean(k1_values)),
        "k1_std": float(np.std(k1_values)),
        "k1_min": float(np.min(k1_values)),
        "k1_max": float(np.max(k1_values)),
        "total_time_seconds": time.time() - t_total,
    }

    print(f"\n[汇总] k1 across {len(seeds)} seeds: mean={summary['k1_mean']:.4f} "
          f"std={summary['k1_std']:.4f} range=[{summary['k1_min']:.4f}, {summary['k1_max']:.4f}]")
    if summary["k1_std"] / max(summary["k1_mean"], 1e-9) > 0.02:
        print("[结论] 不同 seed 收敛到明显不同的 k1 —— 支持论文中'目标函数非凸多峰、"
              "优化器不保证复现同一局部最优'的诊断。")
    else:
        print("[结论] 缩小规模下不同 seed 收敛到接近的 k1 —— 在此规模下未复现原始的"
              "run-to-run 差异，可能是规模缩小改变了目标函数形状 (分辨率/步数都降低)。")

    with open(output_dir / "de_seed_robustness_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[完成] 结果保存至 {output_dir / 'de_seed_robustness_results.json'}")
