import numpy as np
from numba import njit
from scipy.optimize import differential_evolution
from scipy.stats import linregress
import time
import mpmath
import multiprocessing

print("="*85)
print("🚀 终极物理验证：共轭全谱线性拟合 (量子硬件底噪) vs 正半轴纯缩放 (理论极限)")
print("="*85)

# ==========================================
# 1. 精确黎曼零点
# ==========================================
def get_exact_riemann_zeros(n_max=150):
    mpmath.mp.dps = 15
    return np.array([float(mpmath.zetazero(i).imag) for i in range(1, n_max + 1)])

# ==========================================
# 2. 高性能内核：100亿步绝热冷却
# ==========================================
@njit(fastmath=True, nogil=True)
def build_ulam_matrix_anchored(u_c, k_opt, steps, n_bins, offset):
    x = 0.5
    counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    last_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
    
    warmup_steps = 2000000 
    for i in range(warmup_steps):
        u_dyn = u_c + k_opt / (np.log(i + offset)**2)
        x = 1.0 - u_dyn * x**2
        if x > 1.0: x = 0.999
        elif x < -1.0: x = -0.999
            
    for i in range(warmup_steps, steps + warmup_steps):
        u_dyn = u_c + k_opt / (np.log(i + offset)**2)
        x = 1.0 - u_dyn * x**2
        if x > 1.0: x = 0.999
        elif x < -1.0: x = -0.999
        
        current_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
        if 0 <= current_bin < n_bins and 0 <= last_bin < n_bins:
            counts[last_bin, current_bin] += 1
        last_bin = current_bin
        
    return counts

# ==========================================
# 3. 提取全谱特征相位 (绝对自然，不作任何干涉)
# ==========================================
def extract_phases_full_spectrum(counts):
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = counts / row_sums

    vals, _ = np.linalg.eig(P)
    
    # 只要虚部不为 0（即非纯实数解），全部保留！包含正负能量
    valid_all = vals[(np.abs(vals.imag) > 1e-5)]
    
    # 按照自然大小排序（负能量在前，正能量在后）
    return np.unwrap(np.sort(np.angle(valid_all)))

# ==========================================
# 4. 目标函数：双重物理验证体系
# ==========================================
def objective_k_verification(params, target_zeros_pos, u_c, steps, n_bins, offset):
    k_opt = params[0]  
    t_start_time = time.time()
    
    t_end = 1.0 / (np.log(steps + offset)**2)
    u_temp = u_c - k_opt * t_end  
    
    # 执行模拟
    counts = build_ulam_matrix_anchored(u_temp, k_opt, steps, n_bins, offset)
    sys_phases = extract_phases_full_spectrum(counts)
    
    # 将系统相位拆分为正、负两部分
    pos_sys = sys_phases[sys_phases > 0]
    neg_sys = sys_phases[sys_phases < 0]
    
    N_compare = min(len(target_zeros_pos), len(pos_sys), len(neg_sys))
    if N_compare < 80:
        return 1e6
        
    # --------------------------------------------------
    # 【实验 A】: 全谱 + 线性拟合 (模拟量子硬件物理现实)
    # --------------------------------------------------
    # 提取靠近 0 的对称系统谱
    best_neg_sys = neg_sys[-N_compare:] # 最大的几个负数 (如 -0.8, -0.5, -0.1)
    best_pos_sys = pos_sys[:N_compare]  # 最小的几个正数 (如 0.1, 0.5, 0.8)
    aligned_sys_phases = np.concatenate((best_neg_sys, best_pos_sys)) # 严格递增
    
    # 构造绝对对称的黎曼零点目标
    target_neg = -target_zeros_pos[:N_compare][::-1] # (如 -25, -21, -14)
    target_pos = target_zeros_pos[:N_compare]        # (如 14, 21, 25)
    aligned_targets = np.concatenate((target_neg, target_pos)) # 严格递增
    
    # 全谱线性回归
    slope, intercept, r_value, _, _ = linregress(aligned_sys_phases, aligned_targets)
    predicted_full = aligned_sys_phases * slope + intercept
    error_full_linear = np.mean((predicted_full - aligned_targets)**2)
    
    # --------------------------------------------------
    # 【实验 B】: 仅正谱 + 纯缩放 (探究理论数学极限)
    # --------------------------------------------------
    # 强制锚定第一个点，b=0，无视截距和负能量的拉扯
    scale_pure = target_pos[0] / best_pos_sys[0]
    predicted_pos_pure = best_pos_sys * scale_pure
    error_pos_pure = np.mean((predicted_pos_pure - target_pos)**2)
    
    # 打印对比结果，直接用肉眼见证“负能量影子”的代价
    print(f"[Worker] k={k_opt:.4f} | "
          f"物理全谱(含截距) MSE={error_full_linear:.2f} (b={intercept:.3f}) | "
          f"理论正谱(纯缩放) MSE={error_pos_pure:.2f}")
    
    # 我们以“物理全谱”的误差作为优化器的目标，让它去寻找物理现实的最优解
    return error_full_linear

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    u_c = 1.543689
    scan_steps = 10_000_000_000 # 100 亿步，碾压一切噪声
    scan_offset = 100000.0       
    scan_n_bins = 2000           
    
    true_zeros = get_exact_riemann_zeros(100)
    
    print(f"[*] 靶向目标：前100个黎曼零点 (自动映射为 200 个正负对称点)")
    print(f"[*] 正在启动 256核 暴力寻优...")
    
    t_total = time.time()
    
    res = differential_evolution(
        func=objective_k_verification,
        bounds=[(2.0, 10.0)],    
        args=(true_zeros, u_c, scan_steps, scan_n_bins, scan_offset),
        strategy='best1bin',
        maxiter=10,              
        popsize=250,            
        tol=0.01,
        polish=False,           
        workers=-1,             
        disp=True
    )
    
    print(f"\n[+] 寻优结束！总耗时: {(time.time()-t_total)/60:.2f} 分钟")
    print("\n" + "="*70)
    print(f"🎯 终极结论出炉 (全精度锁定版)：")
    # 🔥 核心修改：强制输出 repr() 完整精度！
    print(f"[*] 最优 k1 = {repr(res.x[0])}")
    print(f"[*] 物理现实残差 (全谱 MSE) = {repr(res.fun)}")
    print("="*70)