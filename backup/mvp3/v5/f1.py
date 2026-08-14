import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import multiprocessing as mp
import os
import time

# --- 1. 动力学内核：万亿步演化 (大模型级配置) ---
@njit(fastmath=True, nogil=True)
def scan_kernel(u_c, k, steps, n_bins):
    x = 0.5
    # 热启动
    for _ in range(1000000):
        x = 1 - u_c * x**2
        
    counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    last_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
    
    # 核心演化：结合重整化补偿逻辑初步采样
    for i in range(steps):
        # 引入非自治老化因子（此处为每步的动态耦合 k）
        # 在扫雷阶段，我们保持单次任务内 k 固定，由外部循环分配不同的 k
        x = 1 - (u_c + k / (np.log(i + 2)**2)) * x**2
        current_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
        if 0 <= current_bin < n_bins:
            counts[last_bin, current_bin] += 1
            last_bin = current_bin
    return counts

# --- 2. 单个 Worker 任务 ---
def worker_task(k_val):
    U_C = 1.543689012
    N_BINS = 20000     # 锁定黄金分辨率
    STEPS = 10**10     # 扫雷阶段单点 100 亿步
    TARGET_K = 150     # 每点捕获前 150 个模式，用于趋势对齐
    SAVE_DIR = "riemann_10k_survey"
    
    try:
        t0 = time.time()
        counts = scan_kernel(U_C, k_val, STEPS, N_BINS)
        
        P = csr_matrix(counts)
        row_sums = np.array(P.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        P = P.multiply(1.0 / row_sums[:, np.newaxis])
        
        # 求解特征谱
        vals, _ = eigs(P, k=TARGET_K, which='LM', tol=1e-7)
        phases = np.sort(np.angle(vals[np.abs(vals) > 0.4]))
        
        filename = os.path.join(SAVE_DIR, f"survey_k_{k_val:.4f}.npy")
        np.save(filename, phases)
        return f"[k={k_val:.4f}] Success | Time: {time.time()-t0:.1f}s"
    except Exception as e:
        return f"[k={k_val:.4f}] Error: {str(e)}"

# --- 3. 分布式调度主程序 ---
if __name__ == "__main__":
    SAVE_DIR = "riemann_10k_survey"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    # 定义高维度扫描范围：从红外端(8.5)到紫外端(4.7)
    # 使用 256 个格点，每个核分发一个 k 值任务
    k_range = np.linspace(4.7, 8.5, 256) 
    
    print(f">>> 启动 10,000 点冲击：第一阶段“扫雷”开始")
    print(f">>> 并行规模: 256核 | 步长分布: {k_range[1]-k_range[0]:.4f}")
    
    # 算力全开
    with mp.Pool(processes=256) as pool:
        results = pool.map(worker_task, k_range)
        
    for res in results:
        print(res)
    
    print(">>> 第一阶段扫雷完成，准备进入第二阶段重整化收割。")