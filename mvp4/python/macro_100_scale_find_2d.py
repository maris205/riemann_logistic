import numpy as np
from numba import njit
from scipy.optimize import differential_evolution
import scipy.linalg as la
from scipy.stats import linregress
import time
import mpmath
import multiprocessing

print("="*75)
print("🚀 算力全开：12核并行 2D 全局寻优 (微观 k1 与微观 k2)")
print("="*75)

# ==========================================
# 1. 精确黎曼零点
# ==========================================
def get_exact_riemann_zeros(n_max=150):
    mpmath.mp.dps = 15
    return np.array([float(mpmath.zetazero(i).imag) for i in range(1, n_max + 1)])

# ==========================================
# 2. 高性能内核：引入高阶微扰 k2
# ==========================================
@njit(fastmath=True, nogil=True)
def build_ulam_matrix_2d(u_c, k1, k2, steps, n_bins, offset):
    x = 0.5
    counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    last_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
    
    warmup_steps = 2000000 
    for i in range(warmup_steps):
        # 完整的 2D 微扰对数级数展开
        ln_val = np.log(i + offset)
        u_dyn = u_c + k1 / (ln_val**2) + k2 / (ln_val**3)
        x = 1.0 - u_dyn * x**2
        if x > 1.0: x = 0.999
        elif x < -1.0: x = -0.999
            
    for i in range(warmup_steps, steps + warmup_steps):
        ln_val = np.log(i + offset)
        u_dyn = u_c + k1 / (ln_val**2) + k2 / (ln_val**3)
        x = 1.0 - u_dyn * x**2
        if x > 1.0: x = 0.999
        elif x < -1.0: x = -0.999
        
        current_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
        if 0 <= current_bin < n_bins and 0 <= last_bin < n_bins:
            counts[last_bin, current_bin] += 1
        last_bin = current_bin
        
    return counts

# ==========================================
# 3. 提取特征相位
# ==========================================
def extract_phases_from_matrix(counts):
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = counts / row_sums

    vals, _ = np.linalg.eig(P)
    valid_pos = vals[(np.abs(vals) > 0.4) & (vals.imag > 1e-5)]
    phases = np.unwrap(np.sort(np.angle(valid_pos)))
    return phases

# ==========================================
# 4. 目标函数：支持并行接收向量参数
# ==========================================
def objective_2d(params, target_zeros, u_c, steps, n_bins, offset):
    k1, k2 = params
    t_start = time.time()
    
    # 构建矩阵
    counts = build_ulam_matrix_2d(u_c, k1, k2, steps, n_bins, offset)
    sys_phases = extract_phases_from_matrix(counts)
    
    N_compare = min(len(target_zeros), len(sys_phases))
    if N_compare < 50:
        return 1e6 # 严重惩罚
        
    # 我们不仅对比远场，因为有了 k2 的修正，我们现在可以对比全尺度（包含深水区）
    start_idx = 0  # <--- 核心改变：全场对齐！
    
    slope, intercept, _, _, _ = linregress(sys_phases[:N_compare], target_zeros[:N_compare])
    predicted_zeros = slope * sys_phases + intercept
    
    error = np.mean((predicted_zeros[start_idx:N_compare] - target_zeros[start_idx:N_compare])**2)
    
    t_end = time.time()
    # 注意：在多进程并行时，打印信息可能会交错，这里精简输出
    print(f"[Worker] 测试 k1={k1:.4f}, k2={k2:.4f} | MSE={error:.4f} | 耗时={t_end-t_start:.1f}s")
    return error

# ==========================================
# 5. 主程序：启动 12 核狂飙
# ==========================================
if __name__ == '__main__':
    u_c = 1.543689
    
    scan_steps = 10_000_000_000  # 100 亿步
    scan_offset = 100000.0       
    scan_n_bins = 2000           
    
    print("[*] 正在获取黎曼零点靶向数据...")
    true_zeros = get_exact_riemann_zeros(150)
    
    # 设定 2D 搜索边界
    # k1 依然在之前第一性原理算出来的安全区
    # k2 (1/ln^3 项) 衰减更快，其系数范围可以适当放宽，允许正负补偿
    bounds = [
        (0.01, 0.1),    # k1 的边界
        (-0.5, 0.5)     # k2 的边界
    ]
    
    print(f"[*] 启动全局差分进化寻优 (Differential Evolution)...")
    print(f"[*] 检测到系统 CPU 核心数: {multiprocessing.cpu_count()}")
    print(f"[*] 开启 workers=-1，你的 AutoDL 算力将被彻底榨干！")
    
    t_total = time.time()
    
    # 启动多进程全局寻优
    res = differential_evolution(
        func=objective_2d,
        bounds=bounds,
        args=(true_zeros, u_c, scan_steps, scan_n_bins, scan_offset),
        strategy='best1bin',
        maxiter=10,        # 种群演化代数（不需要太大，10代足以看清趋势）
        popsize=5,         # 种群大小 (每一代评估 popsize * 维数 个样本)
        tol=0.01,
        workers=-1,        # 核心咒语：-1 表示使用所有可用 CPU 核心并行计算
        disp=True          # 每演化完一代打印一次当前最佳结果
    )
    
    print(f"\n[+] 12核算力长跑结束！总耗时: {(time.time()-t_total)/60:.2f} 分钟")
    if res.success:
        best_k1, best_k2 = res.x
        print(f"🎯 成功锁定 2D 全局最优点：")
        print(f"   主 项 微观常数 k1_micro = {best_k1:.6f}")
        print(f"   修正项微观常数 k2_micro = {best_k2:.6f}")
        print(f"[*] 极限谱同构残差 (MSE) = {res.fun:.6f}")
    else:
        print("[-] 演化算法未完全收敛，但已输出过程中发现的最佳解。")