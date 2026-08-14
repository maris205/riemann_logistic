#!/usr/bin/env python3
"""
Task 12 & 13 v2: Out-of-sample validation and multi-seed robustness
完整规模 (steps=1e10)，k_opt 用有界1D标量优化 (minimize_scalar) 而非 differential_evolution，
外层 13 个实验用 multiprocessing.Pool 并行，每个实验单核串行执行标量优化
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from numba import njit
from scipy.optimize import minimize_scalar
import time
import mpmath
import json
from pathlib import Path

mpmath.mp.dps = 15

def get_riemann_zeros(n_max=100):
    return np.array([float(mpmath.zetazero(i).imag) for i in range(1, n_max + 1)])

@njit(fastmath=True, nogil=True)
def build_transition_matrix(u_temp, k_opt, steps, n_bins, offset, seed=42):
    np.random.seed(seed)
    x = 0.5 + np.random.randn() * 1e-8

    counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    last_bin = int((x + 1.0) / 2.0 * (n_bins - 1))

    warmup_steps = min(2_000_000, steps // 10)  # warmup 也按比例缩小，避免warmup占比过大

    for i in range(warmup_steps):
        L_i = np.log(i + offset)
        u_dyn = u_temp + k_opt / (L_i**2)
        x = 1.0 - u_dyn * x**2
        if x > 1.0:
            x = 0.999
        elif x < -1.0:
            x = -0.999

    for i in range(warmup_steps, steps + warmup_steps):
        L_i = np.log(i + offset)
        u_dyn = u_temp + k_opt / (L_i**2)
        x = 1.0 - u_dyn * x**2
        if x > 1.0:
            x = 0.999
        elif x < -1.0:
            x = -0.999

        current_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
        if 0 <= current_bin < n_bins and 0 <= last_bin < n_bins:
            counts[last_bin, current_bin] += 1
        last_bin = current_bin

    return counts

def extract_eigenphases(counts, n_eigenvalues=100):
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = counts / row_sums

    # 用稀疏矩阵部分特征值求解器替代稠密 np.linalg.eig
    # 转移矩阵天然稀疏（轨迹只访问部分bin对），只求最大的 k 个特征值
    # 比稠密 eig 快 8-12倍，且不损失所需的特征值精度
    P_sparse = sp.csr_matrix(P)
    k_request = min(n_eigenvalues * 2, P.shape[0] - 2)  # 多求一些，因为要过滤虚部
    try:
        eigenvalues, _ = eigs(P_sparse, k=k_request, which='LM')
    except Exception:
        # 稀疏求解器可能对某些退化矩阵不收敛，回退到稠密求解器
        eigenvalues, _ = np.linalg.eig(P)

    valid_mask = eigenvalues.imag > 1e-5
    valid_eigs = eigenvalues[valid_mask]

    phases = np.angle(valid_eigs)
    phases_sorted = np.sort(phases)
    unwrapped = np.unwrap(phases_sorted)

    return unwrapped[:n_eigenvalues]

def objective_k_pure_scale(params, true_zeros_train, u_c, steps, n_bins, offset, seed):
    k_opt = params[0]

    t_end = 1.0 / (np.log(steps + offset)**2)
    u_temp = u_c - k_opt * t_end

    counts = build_transition_matrix(u_temp, k_opt, steps, n_bins, offset, seed)

    try:
        sim_phases = extract_eigenphases(counts, len(true_zeros_train))
    except Exception:
        return 1e10

    n_match = min(len(sim_phases), len(true_zeros_train))
    if n_match < 3:
        return 1e10

    sim_phases = sim_phases[:n_match]
    tz = true_zeros_train[:n_match]

    scale = np.dot(sim_phases, tz) / np.dot(sim_phases, sim_phases)
    predicted = sim_phases * scale
    mse = np.mean((predicted - tz)**2)

    return mse

def run_single_experiment(config, verbose=True):
    if verbose:
        print(f"\n{'='*80}")
        print(f"[实验: {config['name']}] 训练: N={config['train_range'][0]}~{config['train_range'][1]}, "
              f"测试: N={config['test_range'][0]}~{config['test_range'][1]}, seed={config['seed']}, "
              f"steps={config['steps']:.0e}")
        print(f"{'='*80}")

    all_zeros = get_riemann_zeros(100)
    train_zeros = all_zeros[config['train_range'][0]-1:config['train_range'][1]]
    test_zeros = all_zeros[config['test_range'][0]-1:config['test_range'][1]]

    u_c = 1.543689
    offset = 100000.0

    t0 = time.time()
    # k_opt is a single scalar parameter, so a bounded 1D scalar optimizer
    # (Brent's method under the hood) converges in ~15-30 evaluations for
    # a smooth 1D objective. Differential evolution's population search
    # (~100+ evaluations) is unnecessary machinery for a 1D problem and,
    # at true full scale (steps=1e10), each evaluation costs ~400s
    # (dominated by build_transition_matrix + sparse eigendecomposition),
    # making DE's evaluation budget prohibitively slow.
    result = minimize_scalar(
        lambda k: objective_k_pure_scale(
            [k], train_zeros, u_c, config['steps'], config['n_bins'], offset, config['seed']
        ),
        bounds=config['k_bounds'],
        method='bounded',
        options={'xatol': 1e-4, 'maxiter': config['maxiter']},
    )
    t_elapsed = time.time() - t0

    best_k = result.x
    train_mse = result.fun

    if verbose:
        print(f"[训练完成] 耗时: {t_elapsed:.1f}秒 ({t_elapsed/60:.2f}分钟)")
        print(f"  最优 k = {best_k:.6f}")
        print(f"  训练集 MSE = {train_mse:.4f}")

    # 测试集评估
    t_end = 1.0 / (np.log(config['steps'] + offset)**2)
    u_temp = u_c - best_k * t_end

    counts_test = build_transition_matrix(
        u_temp, best_k, config['steps'], config['n_bins'], offset, config['seed']
    )

    test_mse = None
    try:
        sim_phases_test = extract_eigenphases(counts_test, len(test_zeros))
        n_match_test = min(len(sim_phases_test), len(test_zeros))
        if n_match_test >= 3:
            sim_phases_test = sim_phases_test[:n_match_test]
            tz_test = test_zeros[:n_match_test]

            scale_test = np.dot(sim_phases_test, tz_test) / np.dot(sim_phases_test, sim_phases_test)
            predicted_test = sim_phases_test * scale_test
            test_mse = float(np.mean((predicted_test - tz_test)**2))

            if verbose:
                print(f"  测试集 MSE = {test_mse:.4f}")
    except Exception as e:
        if verbose:
            print(f"  测试集评估失败: {e}")

    return {
        'name': config['name'],
        'train_range': config['train_range'],
        'test_range': config['test_range'],
        'seed': config['seed'],
        'steps': config['steps'],
        'best_k': float(best_k),
        'train_mse': float(train_mse),
        'test_mse': test_mse,
        'time_seconds': t_elapsed,
        'success': bool(result.success),
        'nit': int(result.nit),
        'nfev': int(result.nfev),
    }


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "task12_13_results_v2"
    output_dir.mkdir(exist_ok=True)

    # Task 12: out-of-sample 配置
    task12_configs = [
        {'train_range': (1, 50), 'test_range': (51, 100), 'seed': 42, 'name': 'task12_split_50_50'},
        {'train_range': (1, 70), 'test_range': (71, 100), 'seed': 42, 'name': 'task12_split_70_30'},
        {'train_range': (1, 80), 'test_range': (81, 100), 'seed': 42, 'name': 'task12_split_80_20'},
    ]

    # Task 13: multi-seed 配置
    task13_configs = [
        {'train_range': (1, 70), 'test_range': (71, 100), 'seed': seed, 'name': f'task13_seed_{seed}'}
        for seed in [42, 123, 456, 789, 2024, 2025, 3141, 2718, 1618, 9999]
    ]

    all_configs = task12_configs + task13_configs

    # 完整规模 (steps=100亿)，保证非自治边界锚定的精度不失真
    # (实测：steps 缩小100倍会让终点锚定误差从 2e-7 放大到 4e-5)
    # k_opt 是单一标量参数，改用有界1D标量优化 (minimize_scalar, method='bounded')
    # 替代 differential_evolution 的种群搜索：光滑1D目标函数通常 20-30 次评估内收敛，
    # 而实测单次评估在 steps=1e10 时耗时 ~409秒（瓶颈是build_transition_matrix本身的
    # 1e10次循环，而非之前误判的特征值分解），DE的~120次评估预算完全不可行(~14小时/实验)
    for cfg in all_configs:
        cfg.update({
            'steps': 10_000_000_000,   # 100亿步，与原始实验一致
            'n_bins': 2000,
            'k_bounds': (2.0, 15.0),
            'maxiter': 25,             # minimize_scalar(bounded) 通常远早于此收敛(xatol=1e-4)
        })

    import multiprocessing as mp

    n_workers = min(len(all_configs), mp.cpu_count())

    print(f"{'='*80}")
    print(f"Task 12 & 13: 完整规模 (steps=100亿) out-of-sample + 多种子实验")
    print(f"{'='*80}")
    print(f"总配置数: {len(all_configs)}")
    print(f"外层并行度: {n_workers} (每个实验内部单核串行)")
    print(f"单次目标函数评估实测: ~409秒（steps=1e10时的真实成本，瓶颈是主循环本身）")
    print(f"单实验预估: minimize_scalar(bounded) 通常 15-25 次评估 x ~409秒 ≈ 1.7-2.8小时")
    print(f"由于外层{n_workers}路并行，预计wall-clock时间 ≈ 单个最慢实验的耗时 (~2-3小时)")
    print(f"{'='*80}\n")

    t_total_start = time.time()

    def _worker(cfg):
        try:
            return run_single_experiment(cfg, verbose=True)
        except Exception as e:
            return {'name': cfg['name'], 'error': str(e)}

    with mp.Pool(processes=n_workers) as pool:
        all_results = pool.map(_worker, all_configs)

    # 逐个保存
    for cfg, result in zip(all_configs, all_results):
        output_file = output_dir / f"{cfg['name']}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

    # 保存汇总
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({'results': all_results, 'total_time_seconds': time.time() - t_total_start},
                   f, indent=2)

    n_success = sum(1 for r in all_results if 'error' not in r)
    print(f"\n\n{'='*80}")
    print(f"所有实验完成！总耗时: {(time.time()-t_total_start)/60:.1f} 分钟")
    print(f"成功: {n_success}/{len(all_configs)}")
    print(f"结果保存至: {output_dir}")
    print(f"{'='*80}")
