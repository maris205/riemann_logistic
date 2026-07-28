import numpy as np
from numba import njit
from scipy.optimize import differential_evolution
import time
import mpmath
import multiprocessing

print("="*85)
print("🚀 算力怪兽重归正道：256核并行 | 强制 b=0 纯比例拟合 (寻找物理圣杯)")
print("="*85)

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

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
def build_ulam_matrix_anchored(u_temp, k_opt, steps, n_bins, offset):
    x = 0.5
    counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    last_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
    
    warmup_steps = 2000000 
    for i in range(warmup_steps):
        L_i = np.log(i + offset)
        u_dyn = u_temp + k_opt / (L_i**2)
        x = 1.0 - u_dyn * x**2
        if x > 1.0: x = 0.999
        elif x < -1.0: x = -0.999
            
    for i in range(warmup_steps, steps + warmup_steps):
        L_i = np.log(i + offset)
        u_dyn = u_temp + k_opt / (L_i**2)
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
def extract_phases(counts):
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = counts / row_sums

    vals, _ = np.linalg.eig(P)
    valid_pos = vals[(vals.imag > 1e-5)]
    return np.unwrap(np.sort(np.angle(valid_pos)))

# ==========================================
# 4. 目标函数：强制 b=0 的线性回归 (仅全局斜率缩放)
# ==========================================
def objective_k_pure_scale(params, target_zeros, u_c, steps, n_bins, offset):
    k_opt = params[0]  
    t_start_time = time.time()
    
    t_end = 1.0 / (np.log(steps + offset)**2)
    u_temp = u_c - k_opt * t_end  
    
    # 执行 100 亿步
    counts = build_ulam_matrix_anchored(u_temp, k_opt, steps, n_bins, offset)
    sys_phases = extract_phases(counts)
    
    N_compare = min(len(target_zeros), len(sys_phases))
    if N_compare < 80:
        return 1e12 # 惩罚项
        
    # 🔥 核心物理法则：强制过原点的最小二乘缩放 (b 锁死为 0)
    # 斜率 a = sum(x*y) / sum(x^2)
    x = sys_phases[:N_compare]
    y = target_zeros[:N_compare]
    slope = np.sum(x * y) / np.sum(x**2)
    
    predicted_zeros = x * slope
    error = np.mean((predicted_zeros - y)**2)
    
    print(f"[Worker] k={k_opt:.6f} | MSE={error:.4f} | Scale={slope:.4f} | 耗时={time.time() - t_start_time:.1f}s")
    return error

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == '__main__':
    u_c = 1.543689
    scan_steps = 10_000_000_000 
    scan_offset = 100000.0           
    scan_n_bins = 2000               
    
    print("[*] 正在加载 100 阶黎曼零点真值...")
    true_zeros = get_exact_riemann_zeros(100)
    
    print(f"[*] 锁定 100 亿步混沌演化 | 物理限制: b ≡ 0 (无漂移)")
    
    t_total = time.time()
    
    res = differential_evolution(
            func=objective_k_pure_scale,
            bounds=[(2.0, 15.0)],    
            args=(true_zeros, u_c, scan_steps, scan_n_bins, scan_offset),
            strategy='best1bin',
            maxiter=15,             
            popsize=256,            
            tol=0.001,
            polish=False,           
            workers=-1,             
            updating='deferred',    
            disp=True
        )
    
    print(f"\n[+] 256核高压寻优结束！总耗时: {(time.time()-t_total)/60:.2f} 分钟")
    
    if res.success or True: 
        best_k = res.x[0]
        
        print("\n" + "圣杯已现 " + "="*50)
        # 🔥 打印 repr(best_k) 确保拿到完整的 16 位浮点数，拒绝任何截断！
        print(f"[*] 最优宏观退火常数 (k1) = {repr(best_k)}")
        print(f"[*] 强制 b=0 的终极极限 MSE = {repr(res.fun)}")
        print("="*60)