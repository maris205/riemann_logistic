import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import mpmath
import time
from numba import njit
from multiprocessing import Pool, cpu_count
import os

# ================== 1. 全局配置与真实零点 ==================
# 提升精度以应对高分辨率扫描
mpmath.mp.dps = 25 

# 🔭 望远镜模式：把 N 往后推！覆盖 N=80 的崩溃区和 N=100+ 的噪声区
N_ZEROS = 200 
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
    try:
        t_start_val = 1.0 / (np.log(1 + C_OFFSET)**2)
        t_end_val   = 1.0 / (np.log(TOTAL_STEPS + C_OFFSET)**2)
        k_opt = DELTA_MU_ABS / (t_start_val - t_end_val)
        u_c = MU_END - k_opt * t_end_val
        
        # 增大 n_bins 以适应更高的 N_ZEROS 解析度
        trans = run_universe_sniper(TOTAL_STEPS, 6000, u_c, k_opt, C_OFFSET, eps)
        
        P_sparse = sp.csr_matrix(trans, dtype=np.float64)
        row_sums = np.array(P_sparse.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0 
        P_sparse.data /= row_sums[P_sparse.indices]
        
        # 求解更多特征值
        # 注意：N_ZEROS 增加后，k 也要相应增加
        eigenvalues, _ = eigs(P_sparse, k=min(N_ZEROS*2 + 50, 5800), which='LM', tol=1e-5)
        
        pos_eigs = eigenvalues[eigenvalues.imag > 1e-4]
        phases = np.sort(np.angle(pos_eigs))
        min_len = min(len(phases), N_ZEROS)
        
        result = {
            'eps': eps,
            'success': False,
            'err_sum': 999.0,
            'msg': f"ε = {eps:<9.6f} | 崩盘 (特征值不足)",
            'zeros_pred': [] # 存储全部预测值以便后续画图
        }

        if min_len >= 6:
            phases_trunc = phases[:min_len]
            true_zeros_trunc = TRUE_ZEROS[:min_len]
            
            scale_factor = TARGETS[0] / phases_trunc[0]
            pred_zeros = phases_trunc * scale_factor
            
            errs = np.abs(pred_zeros[:6] - TARGETS)
            err_sum_2_to_6 = np.sum(errs[1:6])
            
            marker = ""
            if err_sum_2_to_6 < 15.0: marker = "🔥 SUPER!"
            elif err_sum_2_to_6 < 20.0: marker = "✨ Excellent"

            # 格式化输出：保留 6 位小数展示 ε
            msg = f"ε = {eps:<9.6f} | ErrSum: {err_sum_2_to_6:<8.4f} | {marker}"
            if err_sum_2_to_6 < 50.0:
                msg += f"\n  ▶ Z(Top6): {pred_zeros[0]:.2f} | {pred_zeros[1]:.2f}({errs[1]:.2f}) | {pred_zeros[2]:.2f}({errs[2]:.2f}) | {pred_zeros[3]:.2f}({errs[3]:.2f}) | ..."
            
            result['success'] = True
            result['err_sum'] = err_sum_2_to_6
            result['msg'] = msg
            result['zeros_pred'] = pred_zeros # 把这 200 个零点带回来！
            
        return result

    except Exception as e:
        return {
            'eps': eps,
            'success': False,
            'err_sum': 999.0,
            'msg': f"ε = {eps:<9.6f} | 运行出错: {str(e)}",
            'zeros_pred': []
        }

# ================== 4. 主程序入口 ==================
if __name__ == '__main__':
    # 🎯 显微镜模式：加密 10 倍！
    # 步长从 1e-5 变成 1e-6 (0.000001)
    
    # 区间1：宇宙极 (Cosmic Pole)
    eps_cosmic = np.linspace(0.00065, 0.00075, 101) 
    
    # 区间2：实验室极 (Lab Pole) - 这里是重点，加密扫描！
    eps_lab = np.linspace(0.00170, 0.00200, 301)    
    
    eps_array = np.sort(np.concatenate((eps_cosmic, eps_lab)))
    
    num_cores = cpu_count()
    print(f"🚀 启动【256核·饱和式显微镜模式】")
    print(f"🖥️  检测到 CPU 核心数: {num_cores}")
    print(f"📊 扫描任务总数: {len(eps_array)} (步长 1e-6)")
    print(f"🔭 望远镜目标: 预测前 {N_ZEROS} 个零点 (覆盖 N=80 崩溃区)")
    print("=" * 100)
    
    start_total_t = time.time()
    
    best_eps = 0
    min_error_sum = 999.0
    best_zeros = [] # 存储最佳的一组零点
    
    with Pool(processes=num_cores) as pool:
        # 使用 map 并行计算
        results = pool.map(scan_single_eps, eps_array)
        
        for res in results:
            # 只打印“好消息”，避免几百行刷屏
            if res['success'] and res['err_sum'] < 50.0:
                print(res['msg'])
                print("-" * 50)
            
            if res['success'] and res['err_sum'] < min_error_sum:
                min_error_sum = res['err_sum']
                best_eps = res['eps']
                best_zeros = res['zeros_pred']

    print("=" * 100)
    print(f"👑 扫描完成！耗时 {(time.time()-start_total_t):.2f} 秒。")
    print(f"🎯 最终神圣坐标: ε = {best_eps:.6f} (ErrSum = {min_error_sum:.4f})")
    
    # 如果找到了最佳结果，打印出前 20 个看看有没有那几根“大竖线”
    if len(best_zeros) > 20:
        print("\n🔎 最佳参数下的前 20 个预测偏差 (Pred - True):")
        diffs = np.abs(best_zeros[:20] - TRUE_ZEROS[:20])
        for i, diff in enumerate(diffs):
            marker = "❗" if diff > 1.0 else ""
            print(f"  N={i+1:02d}: {diff:.4f} {marker}")