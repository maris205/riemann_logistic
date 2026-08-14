import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import multiprocessing as mp
import os
import time

# --- 1. 加载我们在第一步生成的真值 ---
try:
    TRUE_GAMMAS = np.load("riemann_10k_true.npy")
    print(f">>> 成功加载权威真值，共 {len(TRUE_GAMMAS)} 个零点。")
except:
    print("!!! 错误：未找到 riemann_10k_true.npy，请先运行 gen_truth.py")
    exit()

# --- 2. 动力学内核 (带数值稳定保护) ---
@njit(fastmath=True, nogil=True)
def target_kernel(u_c, k, steps, n_bins):
    x = 0.5
    counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    last_bin = 0
    
    # 200万步热启动，保证概率分布收敛
    for i in range(steps + 2000000):
        # 动态老化项：微观调节
        k_dynamic = k / (np.log(i + 100)**2)
        x = 1 - (u_c + k_dynamic) * x**2
        
        # 数值截断保护
        if x > 1.0: x = 0.999
        if x < -1.0: x = -0.999
            
        current_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
        
        if i > 2000000:
            if 0 <= current_bin < n_bins:
                counts[last_bin, current_bin] += 1
                last_bin = current_bin
    return counts

# --- 3. 狙击手 Worker ---
def sniper_worker(task):
    # 任务包：(分段ID, 起始索引, 结束索引, 物理预测的k值)
    seg_idx, start_n, end_n, k_val = task
    
    U_C = 1.543689012
    N_BINS = 20000
    STEPS = 10**10  # 维持高精度
    SAVE_DIR = "riemann_10k_harvest_pure" # 🌟 改个名字，防覆盖旧数据
    
    try:
        t0 = time.time()
        counts = target_kernel(U_C, k_val, STEPS, N_BINS)
        
        if np.sum(counts) < 1000:
            return f"Seg {seg_idx}: Matrix Empty (k={k_val:.4f})"

        P = csr_matrix(counts)
        row_sums = np.array(P.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        P = P.multiply(1.0 / row_sums[:, np.newaxis])
        
        # 提取目标区间的特征值
        num_targets = end_n - start_n
        # 🌟 核心修改 1：索取量翻倍！因为物理过滤器会扔掉一半的共轭负能量态
        calc_k = num_targets * 2 + 150 
        
        v0 = np.ones(N_BINS) # 初始向量防止 ARPACK error
        vals, _ = eigs(P, k=calc_k, which='LM', tol=1e-4, v0=v0)
        
        # 🌟 核心修改 2：物理过滤器！只保留 虚部 > 0 的纯正能量态
        pure_vals = vals[(np.abs(vals) > 0.4) & (vals.imag > 1e-8)]
        phases = np.sort(np.angle(pure_vals))
        
        # --- 现场比对 (In-flight Validation) ---
        if len(phases) > 0:
            true_segment = TRUE_GAMMAS[start_n:end_n]
            # 简单锚定：用第一个点对齐
            scale = true_segment[0] / phases[0]
            sim_segment = phases[:len(true_segment)] * scale
            
            # 计算这一段的平均误差
            avg_err = np.mean(np.abs(sim_segment - true_segment[:len(sim_segment)]))
        else:
            avg_err = 999.9

        # 保存纯净的相角数据
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR, exist_ok=True)
        filename = os.path.join(SAVE_DIR, f"seg_{seg_idx}_k_{k_val:.4f}_err_{avg_err:.2f}.npy")
        np.save(filename, phases)
        
        return f"Seg {seg_idx} [n={start_n+1}-{end_n}] | k={k_val:.4f} | Err={avg_err:.3f} | {time.time()-t0:.1f}s"
    except Exception as e:
        return f"Seg {seg_idx} ERR: {str(e)}"

# --- 4. 战役指挥部 ---
if __name__ == "__main__":
    SAVE_DIR = "riemann_10k_harvest_pure" # 🌟 指向新目录
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print(f">>> 启动‘万点长征’ (纯正能量版) 收割计划")
    print(f">>> 物理导航公式: k = 4.7 + 10.13/ln(n)")
    
    tasks = []
    total_points = 10000
    segment_size = 100  # 每 100 个点一组，方便并行
    
    for i in range(0, total_points, segment_size):
        start_n = i
        end_n = min(i + segment_size, total_points)
        
        mid_n = (start_n + end_n) / 2 + 1 
        if mid_n < 2: mid_n = 2
        
        k_opt = 4.7000 + 10.13 / np.log(mid_n)
        tasks.append((i//segment_size, start_n, end_n, k_opt))
        
    print(f">>> 任务分发: {len(tasks)} 个分段，覆盖 {total_points} 个零点")
    
    BATCH_SIZE = 48
    total_batches = (len(tasks) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(total_batches):
        batch_tasks = tasks[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        print(f"\n[Batch {i+1}/{total_batches}] 正在收割...")
        
        with mp.Pool(processes=BATCH_SIZE) as pool:
            results = pool.map(sniper_worker, batch_tasks)
            
        for res in results:
            print(f"  {res}")

    print("\n>>> 万点纯正能量收割完成！")