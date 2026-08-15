#!/usr/bin/env python3
"""
Task 26: epsilon (空间离散尺度) 的 out-of-sample 稳定性检验
(回应 58231421 Comment 2 / 58279672 Concern 4)

论文 sec:microscopic-scan / Figure 1 中，最优 epsilon=0.001916 是在全部前 6 个
Riemann 零点 (Z1-Z6) 上联合最小化 ErrSum 找到的 —— 这是同一批数据既用于选择超参数
epsilon，又用于报告拟合质量，存在目标泄漏 (target leakage) 的风险。

这里做一个 train/test 分裂检验：只用其中一部分零点 (train) 去搜索最优 epsilon，
然后看这个 epsilon 在完全没见过的零点 (test) 上的 ErrSum 表现如何，和"作弊"的
全零点联合最优 epsilon 相比是否有明显差距。

出于成本考虑，使用比原始 Fig.1 更小的 steps/n_bins (原始coarse+fine scan合计
在原代码上跑了将近17小时)；这里的目的是方法学一致性检验，不是复现 Fig.1 的
绝对最优值 0.001916。
"""
import numpy as np
from numba import njit
import time
import mpmath
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count

output_dir = Path(__file__).resolve().parent / "task26_results"
output_dir.mkdir(exist_ok=True)


@njit(fastmath=True, nogil=True)
def run_universe_sniper(steps, n_bins, u_c, k_opt, c_offset, eps):
    transitions = np.zeros((n_bins, n_bins), dtype=np.float64)
    V = np.zeros(n_bins, dtype=np.float64)

    dx = 2.0 / n_bins
    init_bin = int((0.5 + 1.0) / dx)
    if init_bin >= n_bins: init_bin = n_bins - 1
    elif init_bin < 0: init_bin = 0
    V[init_bin] = 1.0

    inv_2eps2 = 1.0 / (2.0 * eps**2)
    radius = int(5.0 * eps / dx) + 1

    for n in range(1, steps + 1):
        mu_raw = u_c + k_opt / (np.log(n + c_offset) ** 2.0)
        if mu_raw > 2.0: mu = 2.0
        elif mu_raw < 0.1: mu = 0.1
        else: mu = mu_raw

        V_next = np.zeros(n_bins, dtype=np.float64)
        for i in range(n_bins):
            if V[i] < 1e-12:
                continue
            x = -1.0 + dx * 0.5 + i * dx
            x_next = 1.0 - mu * x * x
            j_center = int((x_next + 1.0) / dx)
            j_start = max(0, j_center - radius)
            j_end = min(n_bins - 1, j_center + radius)

            w_sum = 0.0
            for j in range(j_start, j_end + 1):
                cj = -1.0 + dx * 0.5 + j * dx
                dist_sq = (cj - x_next) * (cj - x_next)
                w_sum += np.exp(-dist_sq * inv_2eps2)

            if w_sum > 1e-18:
                inv_w_sum = 1.0 / w_sum
                for j in range(j_start, j_end + 1):
                    cj = -1.0 + dx * 0.5 + j * dx
                    dist_sq = (cj - x_next) * (cj - x_next)
                    prob = np.exp(-dist_sq * inv_2eps2) * inv_w_sum
                    flow = V[i] * prob
                    V_next[j] += flow
                    transitions[i, j] += flow
            else:
                jc = max(0, min(n_bins - 1, j_center))
                V_next[jc] += V[i]
                transitions[i, jc] += V[i]
        V = V_next
    return transitions


N_ZEROS = 6
TOTAL_STEPS = 200_000
N_BINS = 2000
C_OFFSET = 10.0
MU_END = 1.5437
DELTA_MU_ABS = 0.02


def eval_eps(eps, true_zeros):
    t_start_val = 1.0 / (np.log(1 + C_OFFSET) ** 2)
    t_end_val = 1.0 / (np.log(TOTAL_STEPS + C_OFFSET) ** 2)
    k_opt = DELTA_MU_ABS / (t_start_val - t_end_val)
    u_c = MU_END - k_opt * t_end_val

    import scipy.sparse as sp
    from scipy.sparse.linalg import eigs

    trans = run_universe_sniper(TOTAL_STEPS, N_BINS, u_c, k_opt, C_OFFSET, eps)
    P_sparse = sp.csr_matrix(trans, dtype=np.float64)
    row_sums = np.array(P_sparse.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    P_sparse.data /= row_sums[P_sparse.indices]

    try:
        eigenvalues, _ = eigs(P_sparse, k=min(N_ZEROS * 2 + 20, N_BINS - 2), which="LM", tol=1e-5)
    except Exception:
        return None

    pos_eigs = eigenvalues[eigenvalues.imag > 1e-4]
    phases = np.sort(np.angle(pos_eigs))
    min_len = min(len(phases), N_ZEROS)
    if min_len < 3:
        return None

    phases_trunc = np.unwrap(phases[:min_len])
    true_trunc = true_zeros[:min_len]

    scale = true_trunc[0] / phases_trunc[0] if phases_trunc[0] != 0 else 0.0
    pred = phases_trunc * scale
    abs_err = np.abs(pred - true_trunc)
    return {"eps": float(eps), "n_eigs": int(min_len), "pred": pred.tolist(), "abs_err": abs_err.tolist()}


def _worker(args):
    eps, true_zeros = args
    return eval_eps(eps, true_zeros)


if __name__ == "__main__":
    mpmath.mp.dps = 15
    true_zeros_all = np.array([float(mpmath.zetazero(i).imag) for i in range(1, N_ZEROS + 1)])

    eps_grid = np.geomspace(0.0007, 0.006, 60)

    print(f"{'='*80}")
    print(f"Task 26: epsilon out-of-sample 检验 (steps={TOTAL_STEPS}, n_bins={N_BINS}, "
          f"{len(eps_grid)} 个 eps 候选)")
    print(f"{'='*80}\n")

    n_workers = min(cpu_count(), 32)
    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        raw_results = pool.map(_worker, [(eps, true_zeros_all) for eps in eps_grid])
    print(f"[*] 全部 eps 候选评估完成，耗时 {(time.time()-t0)/60:.1f} 分钟")

    valid = [r for r in raw_results if r is not None]
    print(f"[*] 有效结果数: {len(valid)}/{len(eps_grid)}")

    splits = {
        "train_odd_test_even": {"train_idx": [0, 2, 4], "test_idx": [1, 3, 5]},
        "train_first3_test_last3": {"train_idx": [0, 1, 2], "test_idx": [3, 4, 5]},
    }

    summary = {"eps_grid": eps_grid.tolist(), "raw_results": valid, "splits": {}}

    for split_name, idx in splits.items():
        train_idx, test_idx = idx["train_idx"], idx["test_idx"]
        best_eps_train = None
        best_train_err = np.inf
        for r in valid:
            errs = np.array(r["abs_err"])
            if len(errs) <= max(train_idx):
                continue
            train_err_sum = float(np.sum(errs[train_idx]))
            if train_err_sum < best_train_err:
                best_train_err = train_err_sum
                best_eps_train = r

        if best_eps_train is None:
            summary["splits"][split_name] = {"error": "no valid eps found for this split"}
            continue

        errs = np.array(best_eps_train["abs_err"])
        test_err_sum = float(np.sum(errs[test_idx])) if len(errs) > max(test_idx) else None

        # "cheating" baseline: eps chosen by minimizing error on ALL 6 zeros jointly
        best_all = min(valid, key=lambda r: np.sum(np.array(r["abs_err"])) if len(r["abs_err"]) == N_ZEROS else np.inf)
        errs_all = np.array(best_all["abs_err"])
        test_err_sum_cheat = float(np.sum(errs_all[test_idx])) if len(errs_all) > max(test_idx) else None

        result_split = {
            "train_idx_zerobased": train_idx,
            "test_idx_zerobased": test_idx,
            "best_eps_on_train": best_eps_train["eps"],
            "train_err_sum": best_train_err,
            "test_err_sum_using_train_selected_eps": test_err_sum,
            "joint_all6_optimal_eps": best_all["eps"],
            "test_err_sum_using_joint_optimal_eps": test_err_sum_cheat,
        }
        summary["splits"][split_name] = result_split

        print(f"\n[split: {split_name}]")
        print(f"    train indices (0-based) = {train_idx}, test indices = {test_idx}")
        print(f"    train 上最优 eps = {best_eps_train['eps']:.6f} (train ErrSum={best_train_err:.4f})")
        print(f"    该 eps 在 test 上的 ErrSum = {test_err_sum}")
        print(f"    对照：全部6个零点联合最优 eps = {best_all['eps']:.6f}，"
              f"用它在同样 test 上的 ErrSum = {test_err_sum_cheat}")

    with open(output_dir / "oos_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[完成] 结果保存至 {output_dir / 'oos_results.json'}")
