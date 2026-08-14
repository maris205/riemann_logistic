import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import multiprocessing as mp
import os
import time

# =================================================================
# 1. 核心动力学内核：全标度级数演化 (The Renormalized Kernel)
# =================================================================
@njit(fastmath=True, nogil=True)
def scan_kernel(u_c, k_params, steps, n_bins):
    k0, alpha, beta, gamma = k_params
    x = 0.5
    counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    last_bin = 0 
    
    for i in range(steps + 1000000):
        ln_n = np.log(i + 100)
        # 全量级标度律：k0 + α/ln + β/ln^2 + γ/ln^3
        k_dynamic = k0 + alpha/ln_n + beta/(ln_n**2) + gamma/(ln_n**3)
        
        x = 1 - (u_c + k_dynamic) * x**2
        
        # 边界保护
        if x > 1.0: x = 0.999
        if x < -1.0: x = -0.999
            
        current_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
        
        if i > 1000000:
            if 0 <= current_bin < n_bins:
                counts[last_bin, current_bin] += 1
                last_bin = current_bin
        else:
            last_bin = current_bin
    return counts

# =================================================================
# 2. 算力 Worker：执行单点参数扫描
# =================================================================
def pipeline_worker(k_base):
    U_C = 1.543689012
    N_BINS = 20000     
    STEPS = 5 * 10**9  # 针对高性能服务器，增加采样步数提高物理精度
    TARGET_K = 200     # 提取更多本征值，增加指纹丰富度
    SAVE_DIR = "riemann_10k_survey_full_scale"
    
    # 重整化参数： alpha (1/ln), beta (1/ln^2), gamma (1/ln^3)
    params = (k_base, 0.0, 10.13, 0.5) 
    
    try:
        t0 = time.time()
        counts = scan_kernel(U_C, params, STEPS, N_BINS)
        
        P = csr_matrix(counts)
        row_sums = np.array(P.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        P = P.multiply(1.0 / row_sums[:, np.newaxis])
        
        v0 = np.ones(N_BINS) 
        vals, _ = eigs(P, k=TARGET_K, which='LM', tol=1e-5, v0=v0)
        
        # 提取相角并保存
        phases = np.sort(np.angle(vals[np.abs(vals) > 0.4]))
        
        filename = os.path.join(SAVE_DIR, f"survey_k_{k_base:.4f}.npy")
        np.save(filename, phases)
        
        elapsed = time.time() - t0
        return f"OK_k_{k_base:.4f}_{elapsed:.1f}s"
    except Exception as e:
        return f"ERR_k_{k_base:.4f}_{str(e)}"

# =================================================================
# 3. Main 调度中心：256 核全火力覆盖
# =================================================================
if __name__ == "__main__":
    SAVE_DIR = "riemann_10k_survey_full_scale"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    k_all = np.linspace(4.7, 8.5, 256)
    
    # --- 🌟 针对王博士 256核/480G 内存的暴力配置 🌟 ---
    BATCH_SIZE = 128  # 占用一半物理核心，留出超线程和系统余量，防止带宽瓶颈
    # ----------------------------------------------
    
    print(f"="*60)
    print(f"🚀 启动 256 核怪兽级流水线：全标度渐近演化扫描")
    print(f">>> 物理模型: $k(n) = k_0 + \\beta/\\ln^2(n) + \\gamma/\\ln^3(n)$")
    print(f">>> 并行进程: {BATCH_SIZE} | 剩余核心支持底层矩阵运算")
    print(f"="*60)
    
    total_batches = (len(k_all) + BATCH_SIZE - 1) // BATCH_SIZE
    start_total = time.time()

    for i in range(total_batches):
        batch_ks = k_all[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        print(f"\n[Batch {i+1}/{total_batches}] 正在执行重整化扫描...")
        
        # 使用进程池并行收割
        with mp.Pool(processes=len(batch_ks)) as pool:
            batch_results = pool.map(pipeline_worker, batch_ks)
        
        for res in batch_results:
            print(f"  {res}")
            
    total_elapsed = (time.time() - start_total) / 3600
    print(f"\n" + "="*60)
    print(f"✅ 扫描任务圆满完成！总耗时: {total_elapsed:.2f} 小时")
    print(f">>> 256 组物理指纹已就位，准备进行 Harvest 锁定！")
    print(f"="*60)