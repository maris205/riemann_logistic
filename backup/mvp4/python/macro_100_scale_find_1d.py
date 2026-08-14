import numpy as np
from numba import njit
from scipy.optimize import differential_evolution
import time
import mpmath
import multiprocessing

print("="*80)
print("🚀 算力怪兽觉醒：256核并行 | 全局纯缩放 | 1D 宏观常数 k1 寻优")
print("="*80)

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
        # 提取 log 稍微加速 CPU 计算
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
# 3. 提取特征相位 (解除模长封印)
# ==========================================
def extract_phases(counts):
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = counts / row_sums

    vals, _ = np.linalg.eig(P)
    valid_pos = vals[(vals.imag > 1e-5)]
    return np.unwrap(np.sort(np.angle(valid_pos)))

# ==========================================
# 4. 目标函数：全局纯缩放 MSE 寻优
# ==========================================
def objective_k(params, target_zeros, u_c, steps, n_bins, offset):
    k_opt = params[0]  
    t_start_time = time.time()
    
    t_end = 1.0 / (np.log(steps + offset)**2)
    u_temp = u_c - k_opt * t_end  
    
    # 硬刚 100 亿步
    counts = build_ulam_matrix_anchored(u_temp, k_opt, steps, n_bins, offset)
    sys_phases = extract_phases(counts)
    
    N_compare = min(len(target_zeros), len(sys_phases))
    # 防止因为提取不到足够的特征值而报错
    if N_compare < 80:
        return 1e6
        
    # 🔥 核心修正 2：彻底抛弃 linregress (作弊截距)
    # 强制锚定第 1 个点作为唯一的能量标尺，进行纯比例放大
    scale = target_zeros[0] / sys_phases[0]
    predicted_zeros = sys_phases * scale
    
    # 🔥 核心修正 3：不忽略前 50 个点！计算全频段 (N=1 到 N_compare) 的全局 MSE
    error = np.mean((predicted_zeros[:N_compare] - target_zeros[:N_compare])**2)
    
    print(f"[Worker] 尝试 k={k_opt:.6f} | 纯缩放 Scale={scale:.4f} | 全局 MSE={error:.4f} | 耗时={time.time() - t_start_time:.1f}s")
    return error

# ==========================================
# 5. 主程序：启动 256核 阵列
# ==========================================
if __name__ == '__main__':
    u_c = 1.543689
    scan_steps = 10_000_000_000  # 100 亿步
    scan_offset = 100000.0       
    scan_n_bins = 2000           
    
    print("[*] 正在加载靶向黎曼零点...")
    true_zeros = get_exact_riemann_zeros(100)
    
    print(f"[*] 演化步数已锁定为: 100亿步")
    print(f"[*] 寻优法则: 纯缩放 (Pure Scaling) + 全局误差计算 (不屏蔽前50点)")
    print(f"[*] ⚠️ 正在为 256核 阵列分配任务，起飞！")
    
    t_total = time.time()
    
    # 扩大一点搜索边界，因为之前 7.32 是为了尾部优化的
    res = differential_evolution(
        func=objective_k,
        bounds=[(5.0, 15.0)],    
        args=(true_zeros, u_c, scan_steps, scan_n_bins, scan_offset),
        strategy='best1bin',
        maxiter=10,             
        popsize=250,            # 每一代产生 250 个子代，瞬间喂饱 256 核
        tol=0.01,
        polish=False,           
        workers=-1,             # 榨干所有核心
        updating='deferred',    # 强烈建议加这行，多核寻优时更稳定
        disp=True
    )
    
    print(f"\n[+] 256核暴风寻优结束！总耗时: {(time.time()-t_total)/60:.2f} 分钟")
    if res.success or True: 
        best_k = res.x[0]
        
        print("\n" + "="*70)
        print(f"🎯 算力镇压大获全胜！拿到无死角纯缩放圣杯：")
        # 🔥 核心：用 repr() 输出原汁原味的 64 位机器精度！
        print(f"[*] 最优 1D 宏观退火常数 (k1) = {repr(best_k)}")
        print(f"[*] 全局纯缩放极限残差 (MSE) = {repr(res.fun)}")
        print("="*70)