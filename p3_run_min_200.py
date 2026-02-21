import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import multiprocessing as mp
import time
import os

# --- 1. JIT 极致加速内核 (保持物理逻辑) ---
@njit(fastmath=True, nogil=True)
def compute_matrix_ultra(u_c, k, steps, n_bins):
    n_offset = 1e6
    x = 0.5
    # 热启动进入稳态吸引子
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

# --- 2. 核心任务封装：弹性筛选逻辑 ---
def get_spectrum_task(k_val):
    U_C = 1.543689012  
    STEPS = 10**10     
    N_BINS = 12000     
    TARGET_MODES = 200 
    SAVE_DIR = "riemann_200_pure"
    
    save_path = os.path.join(SAVE_DIR, f"pure_res_k_{k_val:.4f}_steps10t10.npy")
    if os.path.exists(save_path): 
        return f"Skip: {k_val:.4f}"
    
    t0 = time.time()
    try:
        # 执行暴力演化
        counts = compute_matrix_ultra(U_C, k_val, STEPS, N_BINS)
        
        # 算子构建
        P = csr_matrix(counts)
        row_sums = np.array(P.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        P = P.multiply(1.0 / row_sums[:, np.newaxis])
        
        # 索要更多特征值 (k=300)，并放宽过滤门槛至 0.4
        vals, _ = eigs(P, k=300, which='LM', tol=1e-7)
        phases = np.sort(np.angle(vals[np.abs(vals) > 0.4])) 
        phases = phases[phases > 0.05]
        
        # --- 弹性保存逻辑：规避 ValueError ---
        actual_len = len(phases)
        if actual_len >= TARGET_MODES:
            # 鱼够多，取前 200 个规整化
            final_data = np.array(phases[:TARGET_MODES], dtype=np.float64)
            np.save(save_path, final_data)
            msg = f"[Done] k={k_val:.4f} | Got {TARGET_MODES}/200"
        else:
            # 鱼不够，有多少存多少，标记为 Partial
            final_data = np.array(phases, dtype=np.float64)
            np.save(save_path, final_data)
            msg = f"[Partial] k={k_val:.4f} | Found {actual_len} points (Collapsed)"
            
    except Exception as e:
        msg = f"[Error] k={k_val:.4f} | {str(e)}"
    
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
    
    print(f"弹性重挖启动 | 目标: 200点(尽量) | 门槛: 0.4 | 步数: 10^10")
    
    with mp.Pool(processes=64) as pool:
        pool.map(get_spectrum_task, k_range)

    print("\n>>> 数据收割中，请继续看剧，脚本会自动处理异常情况。")