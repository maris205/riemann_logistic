import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import multiprocessing as mp
import os
import time

# --- 1. 核心改进：带边界保护的动力学内核 ---
@njit(fastmath=True, nogil=True)
def scan_kernel(u_c, k, steps, n_bins):
    x = 0.5
    counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    last_bin = 0 # 初始初始化
    
    # 模拟演化过程
    for i in range(steps + 1000000):
        # 平滑老化项：将起始偏移设为 100，防止 i 较小时 k 冲量过大
        k_dynamic = k / (np.log(i + 100)**2)
        x = 1 - (u_c + k_dynamic) * x**2
        
        # 边界截断保护 (Clipping)：防止数值飞出相空间导致 Empty Matrix
        if x > 1.0: x = 0.999
        if x < -1.0: x = -0.999
            
        current_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
        
        # 1,000,000 步之后开始构建转移矩阵
        if i > 1000000:
            if 0 <= current_bin < n_bins:
                # 记录状态转移 counts[from, to]
                counts[last_bin, current_bin] += 1
                last_bin = current_bin
        else:
            # 热启动阶段只更新 last_bin
            last_bin = current_bin
            
    return counts

# --- 2. 增强型 Worker ---
def pipeline_worker(k_val):
    U_C = 1.543689012
    N_BINS = 20000     
    STEPS = 10**10     # 扫雷维持 100 亿步量级
    TARGET_K = 150     
    SAVE_DIR = "riemann_10k_survey"
    
    try:
        t0 = time.time()
        # 调用数值稳定的内核
        counts = scan_kernel(U_C, k_val, STEPS, N_BINS)
        
        # 矩阵非空校验
        if np.sum(counts) < 100:
            return f"ERR_k_{k_val:.4f}_Insufficient_Data"

        P = csr_matrix(counts)
        row_sums = np.array(P.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        P = P.multiply(1.0 / row_sums[:, np.newaxis])
        
        # 初始向量引导，彻底解决 ARPACK -9
        v0 = np.ones(N_BINS) 
        vals, _ = eigs(P, k=TARGET_K, which='LM', tol=1e-4, v0=v0)
        
        phases = np.sort(np.angle(vals[np.abs(vals) > 0.4]))
        
        filename = os.path.join(SAVE_DIR, f"survey_k_{k_val:.4f}.npy")
        np.save(filename, phases)
        return f"OK_k_{k_val:.4f}_{time.time()-t0:.1f}s"
    except Exception as e:
        return f"ERR_k_{k_val:.4f}_{str(e)}"

# --- 3. 流水线调度 ---
if __name__ == "__main__":
    SAVE_DIR = "riemann_10k_survey"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 扫描 256 个点
    k_all = np.linspace(4.7, 8.5, 256)
    BATCH_SIZE = 48  # 考虑内存带宽平衡
    
    print(f">>> 启动 10k 冲击第一阶段：数值稳定版流水线")
    print(f">>> 保护机制：数值截断 + 初始向量引导")
    
    total_batches = (len(k_all) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(total_batches):
        batch_ks = k_all[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        print(f"\n[Batch {i+1}/{total_batches}] 正在处理...")
        
        with mp.Pool(processes=BATCH_SIZE) as pool:
            batch_results = pool.map(pipeline_worker, batch_ks)
        
        for res in batch_results:
            print(f"  {res}")
            
    print("\n>>> 全量扫雷任务已完成，请检查 riemann_10k_survey 目录。")