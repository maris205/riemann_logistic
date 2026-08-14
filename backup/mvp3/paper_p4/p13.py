import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import mpmath
import time
from numba import njit
from multiprocessing import Pool, cpu_count, current_process
import os

# ================== 1. 全局配置与真实零点 ==================
# 这些变量在 Linux (AutoDL) 的 fork 模式下会被子进程继承，无需重复计算
mpmath.mp.dps = 15
N_ZEROS = 100
TRUE_ZEROS = np.array([float(mpmath.zetazero(i).imag) for i in range(1, N_ZEROS + 1)])
TARGETS = TRUE_ZEROS[:6]

# 任务参数
MU_END = 1.5437
TOTAL_STEPS = 10**6     
C_OFFSET = 10.0    
DELTA_MU_ABS = 0.02    

# ================== 2. 高斯核概率溅射引擎 (Numba 加速) ==================
@njit
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
        # 1/ln^2 动力学律
        mu_raw = u_c + k_opt / (np.log(n + c_offset)**2.0)
        
        if mu_raw > 2.0: mu = 2.0
        elif mu_raw < 0.1: mu = 0.1
        else: mu = mu_raw
            
        V_next = np.zeros(n_bins, dtype=np.float64)
        
        for i in range(n_bins):
            if V[i] < 1e-12: continue
                
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
                if j_center < 0: j_center = 0
                if j_center >= n_bins: j_center = n_bins - 1
                flow = V[i]
                V_next[j_center] += flow
                transitions[i, j_center] += flow
                
        V = V_next 
        
    return transitions

# ================== 3. 单个 EPS 的处理函数 (Worker) ==================
def scan_single_eps(eps):
    """
    这是每个 CPU 核心独立运行的函数
    """
    try:
        # 独立的参数计算
        t_start_val = 1.0 / (np.log(1 + C_OFFSET)**2)
        t_end_val   = 1.0 / (np.log(TOTAL_STEPS + C_OFFSET)**2)
        k_opt = DELTA_MU_ABS / (t_start_val - t_end_val)
        u_c = MU_END - k_opt * t_end_val
        
        # 运行模拟
        trans = run_universe_sniper(TOTAL_STEPS, 5000, u_c, k_opt, C_OFFSET, eps)
        
        # 稀疏矩阵处理
        P_sparse = sp.csr_matrix(trans, dtype=np.float64)
        row_sums = np.array(P_sparse.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0 
        P_sparse.data /= row_sums[P_sparse.indices]
        
        # 求解特征值
        eigenvalues, _ = eigs(P_sparse, k=N_ZEROS*2 + 20, which='LM', tol=1e-4)
        
        # 筛选正向频率
        pos_eigs = eigenvalues[eigenvalues.imag > 1e-4]
        phases = np.sort(np.angle(pos_eigs))
        min_len = min(len(phases), N_ZEROS)
        
        result = {
            'eps': eps,
            'success': False,
            'err_sum': 999.0,
            'msg': f"ε = {eps:<8.5f} | 崩盘 (特征值不足)"
        }

        if min_len >= 6:
            phases_trunc = phases[:min_len]
            true_zeros_trunc = TRUE_ZEROS[:min_len]
            
            # 比例缩放
            scale_factor = TARGETS[0] / phases_trunc[0]
            pred_zeros = phases_trunc * scale_factor
            
            errs = np.abs(pred_zeros[:6] - TARGETS)
            err_sum_2_to_6 = np.sum(errs[1:6])
            
            # 格式化输出信息
            marker = ""
            if err_sum_2_to_6 < 20.0: marker = "✨ Excellent"
            if err_sum_2_to_6 < 15.0: marker = "🔥 SUPER!"

            msg = f"ε = {eps:<8.5f} | ErrSum: {err_sum_2_to_6:<8.4f} | {marker}"
            if err_sum_2_to_6 < 100.0:
                msg += f"\n  ▶ Z: {pred_zeros[0]:.2f} | {pred_zeros[1]:.2f}({errs[1]:.2f}) | {pred_zeros[2]:.2f}({errs[2]:.2f}) | {pred_zeros[3]:.2f}({errs[3]:.2f}) | {pred_zeros[4]:.2f}({errs[4]:.2f}) | {pred_zeros[5]:.2f}({errs[5]:.2f})"
            
            result['success'] = True
            result['err_sum'] = err_sum_2_to_6
            result['msg'] = msg
            
        return result

    except Exception as e:
        return {
            'eps': eps,
            'success': False,
            'err_sum': 999.0,
            'msg': f"ε = {eps:<8.5f} | 运行出错: {str(e)}"
        }

# ================== 4. 主程序入口 ==================
if __name__ == '__main__':
    # 🎯 定义扫描区间
    # 区间1：宇宙极 (Cosmic Pole) ~ 0.00070
    eps_cosmic = np.linspace(0.00065, 0.00075, 11) # 0.00001 步长
    # 区间2：实验室极 (Lab Pole) ~ 0.00183
    eps_lab = np.linspace(0.00170, 0.00200, 31)    # 0.00001 步长
    
    # 合并任务列表
    eps_array = np.sort(np.concatenate((eps_cosmic, eps_lab)))
    
    # 获取核心数（AutoDL 容器内通常能正确获取）
    num_cores = cpu_count()
    print(f"🚀 启动【多核显微镜模式】")
    print(f"🖥️  检测到 CPU 核心数: {num_cores}")
    print(f"📊 扫描任务总数: {len(eps_array)}")
    print(f"🎯 目标区间: [0.00065-0.00075] & [0.00170-0.00200]")
    print("=" * 100)
    
    start_total_t = time.time()
    
    best_eps = 0
    min_error_sum = 999.0
    
    # ⚡️ 建立进程池，并行执行
    with Pool(processes=num_cores) as pool:
        # map 会按顺序返回结果，但在计算时是并行的
        # 如果想看到实时乱序输出，可以用 imap_unordered，这里为了逻辑简单用 map
        results = pool.map(scan_single_eps, eps_array)
        
        # 处理结果
        for res in results:
            print(res['msg'])
            print("-" * 50)
            
            if res['success'] and res['err_sum'] < min_error_sum:
                min_error_sum = res['err_sum']
                best_eps = res['eps']

    print("=" * 100)
    print(f"👑 并行扫描完成！耗时 {(time.time()-start_total_t)/60:.2f} 分钟。")
    print(f"🎯 最终神圣坐标: ε = {best_eps:.5f} (ErrSum = {min_error_sum:.4f})")