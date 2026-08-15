#!/usr/bin/env python3
"""
Task 25: Gaussian-splatting 与 Monte-Carlo 轨迹构造在共享 N 范围上的一致性检验
(回应 58231421 Comment 5)

固定完全相同的物理参数 (u_c, k1, offset, 总步数, n_bins)，仅改变转移矩阵的构造方式:
  (A) Monte Carlo 轨迹硬计数 (macro 方法): 单一轨迹逐步演化，落入哪个 bin 就给那个 bin 记一次。
  (B) Gaussian 核概率溅射 (micro 方法): 整个概率密度分布逐步演化，每步把质量按高斯核
      散布到目标邻域的多个 bin。

注意：Gaussian 溅射方法单步成本是 O(n_bins * radius)，远高于 MC 硬计数的 O(1)，
因此这里两种方法使用相同的总步数，但受限于溅射方法的成本，步数远小于宏观论文里的 1e10。
这是一个方法学一致性检验（两种构造在同一物理系统上是否给出统计一致的特征相位），
不是对宏观 k1=6.481/8.070 那个大规模数值结果的复现。
"""
import numpy as np
from numba import njit
import time
import mpmath
import json
from pathlib import Path

output_dir = Path(__file__).resolve().parent / "task25_results"
output_dir.mkdir(exist_ok=True)


def get_exact_riemann_zeros(n_max=20):
    mpmath.mp.dps = 15
    return np.array([float(mpmath.zetazero(i).imag) for i in range(1, n_max + 1)])


@njit(fastmath=True, nogil=True)
def build_mc_matrix(u_c, k1, steps, n_bins, offset):
    x = 0.5
    counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    last_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
    warmup = min(200_000, steps // 5)
    for i in range(warmup):
        L_i = np.log(i + offset)
        u_dyn = u_c + k1 / (L_i**2)
        x = 1.0 - u_dyn * x**2
        if x > 1.0: x = 0.999
        elif x < -1.0: x = -0.999
    for i in range(warmup, steps + warmup):
        L_i = np.log(i + offset)
        u_dyn = u_c + k1 / (L_i**2)
        x = 1.0 - u_dyn * x**2
        if x > 1.0: x = 0.999
        elif x < -1.0: x = -0.999
        current_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
        if 0 <= current_bin < n_bins and 0 <= last_bin < n_bins:
            counts[last_bin, current_bin] += 1
        last_bin = current_bin
    return counts


@njit(fastmath=True, nogil=True)
def build_gaussian_splatting_matrix(u_c, k1, steps, n_bins, offset, eps):
    transitions = np.zeros((n_bins, n_bins), dtype=np.float64)
    V = np.zeros(n_bins, dtype=np.float64)
    dx = 2.0 / n_bins
    init_bin = int((0.5 + 1.0) / dx)
    if init_bin >= n_bins: init_bin = n_bins - 1
    elif init_bin < 0: init_bin = 0
    V[init_bin] = 1.0

    inv_2eps2 = 1.0 / (2.0 * eps**2)
    radius = int(5.0 * eps / dx) + 1

    warmup = min(200_000, steps // 5)
    for n in range(1, warmup + steps + 1):
        L_i = np.log(n + offset)
        mu_raw = u_c + k1 / (L_i**2)
        mu = max(0.1, min(2.0, mu_raw))

        V_next = np.zeros(n_bins, dtype=np.float64)
        for i in range(n_bins):
            if V[i] < 1e-14:
                continue
            xi = -1.0 + dx * 0.5 + i * dx
            x_next = 1.0 - mu * xi * xi
            j_center = int((x_next + 1.0) / dx)
            j_start = max(0, j_center - radius)
            j_end = min(n_bins - 1, j_center + radius)

            w_sum = 0.0
            for j in range(j_start, j_end + 1):
                cj = -1.0 + dx * 0.5 + j * dx
                dist_sq = (cj - x_next) * (cj - x_next)
                w_sum += np.exp(-dist_sq * inv_2eps2)

            if w_sum > 1e-18:
                inv_w = 1.0 / w_sum
                for j in range(j_start, j_end + 1):
                    cj = -1.0 + dx * 0.5 + j * dx
                    dist_sq = (cj - x_next) * (cj - x_next)
                    prob = np.exp(-dist_sq * inv_2eps2) * inv_w
                    flow = V[i] * prob
                    V_next[j] += flow
                    if n > warmup:
                        transitions[i, j] += flow
            else:
                jc = max(0, min(n_bins - 1, j_center))
                V_next[jc] += V[i]
                if n > warmup:
                    transitions[i, jc] += V[i]
        V = V_next
    return transitions


def extract_phases_module_c(counts, cutoff=0.4):
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = counts / row_sums
    vals, _ = np.linalg.eig(P)
    mask = (vals.imag > 1e-5) & (np.abs(vals) > cutoff)
    sub = vals[mask]
    phases = np.sort(np.angle(sub))
    return np.unwrap(phases)


def forced_origin_fit(phases, true_zeros):
    n = min(len(phases), len(true_zeros))
    if n < 3:
        return None
    x = phases[:n]
    y = true_zeros[:n]
    slope = np.sum(x * y) / np.sum(x**2)
    pred = x * slope
    mse = float(np.mean((pred - y) ** 2))
    return {"n_used": int(n), "slope": float(slope), "mse": mse}


if __name__ == "__main__":
    u_c = 1.543689
    k1 = 6.481
    offset = 100000.0
    N_ZEROS = 20

    steps = 200_000
    n_bins = 1000
    eps = 0.01

    true_zeros = get_exact_riemann_zeros(N_ZEROS)

    print(f"{'='*80}")
    print("Task 25: Gaussian-splatting vs Monte-Carlo 一致性检验")
    print(f"共同参数: u_c={u_c}, k1={k1}, offset={offset}, steps={steps}, n_bins={n_bins}")
    print(f"{'='*80}\n")

    print("[A] 构造 Monte Carlo 硬计数矩阵...")
    t0 = time.time()
    counts_mc = build_mc_matrix(u_c, k1, steps, n_bins, offset)
    t_mc = time.time() - t0
    print(f"    完成，耗时 {t_mc:.1f}s")

    print(f"[B] 构造 Gaussian 溅射矩阵 (eps={eps})...")
    t0 = time.time()
    counts_gs = build_gaussian_splatting_matrix(u_c, k1, steps, n_bins, offset, eps)
    t_gs = time.time() - t0
    print(f"    完成，耗时 {t_gs:.1f}s")

    print("\n[*] 对两份矩阵分别提取特征相位 (Module C 标准流程: |lambda|>0.4, 上半平面, 排序+展开)...")
    phases_mc = extract_phases_module_c(counts_mc)
    phases_gs = extract_phases_module_c(counts_gs)
    print(f"    MC 保留特征值数: {len(phases_mc)}")
    print(f"    Gaussian splatting 保留特征值数: {len(phases_gs)}")

    n_shared = min(len(phases_mc), len(phases_gs), N_ZEROS)
    print(f"\n[*] 共享比较范围: N=1..{n_shared}")

    result = {
        "params": {"u_c": u_c, "k1": k1, "offset": offset, "steps": steps,
                    "n_bins": n_bins, "eps": eps, "N_ZEROS": N_ZEROS},
        "n_eigs_mc": int(len(phases_mc)),
        "n_eigs_gaussian_splatting": int(len(phases_gs)),
        "n_shared": int(n_shared),
        "build_time_seconds": {"mc": t_mc, "gaussian_splatting": t_gs},
    }

    if n_shared >= 3:
        pm = phases_mc[:n_shared]
        pg = phases_gs[:n_shared]

        # 直接比较两组原始（未各自拟合）的展开相位序列
        corr = float(np.corrcoef(pm, pg)[0, 1]) if n_shared >= 2 else None
        rms_ratio = float(np.sqrt(np.mean((pm - pg) ** 2)) / (np.mean(np.abs(pm)) + 1e-12))

        print(f"\n[*] 原始展开相位序列直接比较 (无拟合):")
        print(f"    Pearson correlation(theta_MC, theta_GaussianSplatting) = {corr}")
        print(f"    RMS(theta_MC - theta_GS) / mean(|theta_MC|) = {rms_ratio:.4f}")

        fit_mc = forced_origin_fit(pm, true_zeros)
        fit_gs = forced_origin_fit(pg, true_zeros)
        print(f"\n[*] 各自独立强制过原点拟合黎曼零点 (N={n_shared}):")
        print(f"    MC:                  slope={fit_mc['slope']:.4f}  MSE={fit_mc['mse']:.4f}")
        print(f"    Gaussian splatting:  slope={fit_gs['slope']:.4f}  MSE={fit_gs['mse']:.4f}")

        result.update({
            "raw_phase_correlation": corr,
            "raw_phase_rms_relative_diff": rms_ratio,
            "fit_mc": fit_mc,
            "fit_gaussian_splatting": fit_gs,
            "theta_mc": pm.tolist(),
            "theta_gaussian_splatting": pg.tolist(),
        })
    else:
        print("[!] 共享比较范围内特征值数量不足 (<3)，无法定量比较。")
        result["error"] = "insufficient shared eigenvalues"

    with open(output_dir / "overlap_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[完成] 结果保存至 {output_dir / 'overlap_results.json'}")
