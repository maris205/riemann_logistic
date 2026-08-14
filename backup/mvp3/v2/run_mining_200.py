import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import multiprocessing as mp
import time
import os

# --- 1. JIT 极致加速内核 (保持你的核心物理逻辑不变) ---
@njit(fastmath=True, nogil=True)
def compute_matrix_ultra(u_c, k, steps, n_bins):
    n_offset = 1e6
    x = 0.5
    # 充分热启动
    for i in range(500000):
        u = u_c - k * (np.log(i + n_offset))**-2
        x = 1 - u * x**2
    
    counts = np.zeros((n_bins, n_bins), dtype=np.float64) 
    last_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
    
    for i in range(steps):
        u = u_c - k * (np.log(i + n_offset))**-2
        x = 1 - u * x**2
        current_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
        if 0 <= current_bin < n_bins:
            counts[last_bin, current_bin] += 1
            last_bin = current_bin
    return counts

# --- 2. 封装 Worker 任务：加入严格 Shape 检查 ---
def get_spectrum_task(k_val):
    # --- 统一规格定义 ---
    U_C = 1.543689012  
    STEPS = 10**10     
    N_BINS = 12000     
    TARGET_MODES = 200 # 明确目标点数
    SAVE_DIR = "riemann_200_pure"
    
    save_path = os.path.join(SAVE_DIR, f"pure_res_k_{k_val:.4f}_steps10t10.npy")
    
    # 幂等性检查：支持看剧期间的断点续传
    if os.path.exists(save_path):
        return f"Skip: {k_val:.4f}"
    
    t0 = time.time()
    try:
        # 执行动力学演化
        counts = compute_matrix_ultra(U_C, k_val, STEPS, N_BINS)
        
        # 算子构建与特征提取
        P = csr_matrix(counts)
        row_sums = np.array(P.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        P = P.multiply(1.0 / row_sums[:, np.newaxis])
        
        # 提取前 200+10 个主模式，确保筛选后仍够 200 个
        vals, _ = eigs(P, k=TARGET_MODES + 10, which='LM', tol=1e-7)
        phases = np.sort(np.angle(vals[np.abs(vals) > 0.6]))
        phases = phases[phases > 0.05]
        
        # --- 核心：严格规格对齐 ---
        if len(phases) >= TARGET_MODES:
            # 强制截取前 200 个点，确保 .npy 文件 shape 永远是 (200,)
            final_data = np.array(phases[:TARGET_MODES], dtype=np.float64)
            np.save(save_path, final_data)
            msg = f"[Done] k={k_val:.4f} | Time: {time.time()-t0:.1f}s | Path: {save_path}"
        else:
            msg = f"[Fail] k={k_val:.4f} Insufficient points ({len(phases)}/200)"
            
    except Exception as e:
        msg = f"[Error] k={k_val:.4f} | Exception: {str(e)}"
    
    print(msg)
    return msg

# --- 3. 256 核全自动调度 ---
if __name__ == "__main__":
    SAVE_DIR = "riemann_200_pure"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f">>> 创建纯净目录: {SAVE_DIR} [cite: 2026-02-10]")

    # 扫描 64 个 k 值
    k_range = np.linspace(4.7, 12.73, 64) 
    
    print(f"暴力重挖启动 | 配置: 256核心/480GB | 目标模式: 200 | 步数: 10^10")
    
    # AutoDL 进程池：设为 64 以平衡百亿步迭代和特征值求解的负载
    with mp.Pool(processes=64) as pool:
        pool.map(get_spectrum_task, k_range)

    print("\n>>> 200点纯净数据集构建完成。你可以继续看剧了。")