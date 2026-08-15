#!/usr/bin/env python3
"""
Task 24: Module C 相位提取的稳健性检验（回应 58231636 Comment 10 / 58279672 Concern 5）

固定 k1=6.481 (Table 1 Strategy C 的数值)，用完整规模 (steps=1e10) 构造一次经验转移矩阵，
然后在这一份特征值谱上，独立改变三类 Module C 的方法学选择，观察拟合结果 (slope, MSE) 的
稳定性:
  (a) magnitude cutoff |lambda| 的阈值: 0.2 / 0.3 / 0.4(论文用值) / 0.5 / 0.6
  (b) 共轭对分支选择: 上半平面(论文用) / 下半平面 / 用 abs(角度) 折叠两半平面
  (c) 排序后展开(论文用) vs 不排序直接展开(按 eig 返回的原始顺序)

矩阵只构造一次 (最贵的部分)，后续所有敏感性分支都在同一份特征值上做后处理，成本很低。
"""
import numpy as np
from numba import njit
import time
import mpmath
import json
from pathlib import Path

output_dir = Path(__file__).resolve().parent / "task24_results"
output_dir.mkdir(exist_ok=True)


def get_exact_riemann_zeros(n_max=100):
    mpmath.mp.dps = 15
    return np.array([float(mpmath.zetazero(i).imag) for i in range(1, n_max + 1)])


@njit(fastmath=True, nogil=True)
def build_ulam_matrix_anchored(u_temp, k_opt, steps, n_bins, offset):
    x = 0.5
    counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    last_bin = int((x + 1.0) / 2.0 * (n_bins - 1))

    warmup_steps = 2000000
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


def fit_forced_origin(phases, true_zeros):
    n = min(len(phases), len(true_zeros))
    if n < 10:
        return None
    x = phases[:n]
    y = true_zeros[:n]
    slope = np.sum(x * y) / np.sum(x**2)
    pred = x * slope
    mse = float(np.mean((pred - y) ** 2))
    rel_err = float(np.mean(np.abs(pred - y) / y))
    return {"n_used": int(n), "slope": float(slope), "mse": mse, "mean_rel_err": rel_err}


def extract_variant(vals, cutoff, branch, order):
    if branch == "upper":
        mask = (vals.imag > 1e-5) & (np.abs(vals) > cutoff)
    elif branch == "lower":
        mask = (vals.imag < -1e-5) & (np.abs(vals) > cutoff)
    elif branch == "folded_abs":
        mask = (np.abs(vals.imag) > 1e-5) & (np.abs(vals) > cutoff)
    else:
        raise ValueError(branch)

    sub = vals[mask]
    if branch == "folded_abs":
        angles = np.abs(np.angle(sub))
    else:
        angles = np.angle(sub)

    if order == "sorted":
        angles = np.sort(angles)
    # order == "raw": keep eig()'s native return order, no sort

    return np.unwrap(angles)


if __name__ == "__main__":
    u_c = 1.543689
    k1_fixed = 6.481  # Table 1, Strategy C
    steps = 10_000_000_000
    offset = 100000.0
    n_bins = 2000

    true_zeros = get_exact_riemann_zeros(100)

    t_end = 1.0 / (np.log(steps + offset) ** 2)
    u_temp = u_c - k1_fixed * t_end

    print(f"[*] 构造经验转移矩阵一次 (k1={k1_fixed}, steps={steps:.0e})...")
    t0 = time.time()
    counts = build_ulam_matrix_anchored(u_temp, k1_fixed, steps, n_bins, offset)
    print(f"[*] 矩阵构造完成，耗时 {time.time()-t0:.1f}s")

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = counts / row_sums

    print("[*] 对同一份 P 做一次稠密特征值分解...")
    t0 = time.time()
    vals, _ = np.linalg.eig(P)
    print(f"[*] 特征值分解完成，耗时 {time.time()-t0:.1f}s, 共 {len(vals)} 个特征值")

    cutoffs = [0.2, 0.3, 0.4, 0.5, 0.6]
    branches = ["upper", "lower", "folded_abs"]
    orders = ["sorted", "raw"]

    results = []
    baseline = None
    for cutoff in cutoffs:
        for branch in branches:
            for order in orders:
                try:
                    phases = extract_variant(vals, cutoff, branch, order)
                    fit = fit_forced_origin(phases, true_zeros)
                except Exception as e:
                    fit = None
                row = {
                    "cutoff": cutoff, "branch": branch, "order": order,
                    "n_eigs_kept": int(len(phases)) if fit is not None else 0,
                    "fit": fit,
                }
                results.append(row)
                tag = "  <-- paper baseline" if (cutoff == 0.4 and branch == "upper" and order == "sorted") else ""
                if fit:
                    print(f"cutoff={cutoff:.1f} branch={branch:10s} order={order:6s} "
                          f"n_used={fit['n_used']:3d} slope={fit['slope']:.4f} "
                          f"MSE={fit['mse']:.3f} mean_rel_err={fit['mean_rel_err']*100:.2f}%{tag}")
                else:
                    print(f"cutoff={cutoff:.1f} branch={branch:10s} order={order:6s} FAILED (too few eigenvalues){tag}")
                if cutoff == 0.4 and branch == "upper" and order == "sorted":
                    baseline = fit

    out = {
        "k1_fixed": k1_fixed,
        "steps": steps,
        "paper_baseline_cutoff_0.4_upper_sorted": baseline,
        "all_variants": results,
    }
    with open(output_dir / "sensitivity_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[完成] 结果保存至 {output_dir / 'sensitivity_results.json'}")
