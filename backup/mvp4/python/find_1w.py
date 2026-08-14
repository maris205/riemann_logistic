import os
# 🔥 核心战术：16个并发Worker，每个Worker调用12个底层BLAS线程，榨干 256核 的矩阵加速能力！
# 必须在导入 numpy 前设置
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "12"

import numpy as np
from numba import njit
from scipy.optimize import differential_evolution
import scipy.linalg  # 强制使用 scipy.linalg 获取 sgeev 原生单精度支持
import time
import multiprocessing

# ================= 1. 日志系统 =================
LOG_FILE = "macro_1d_10k_limit.log"
def log_msg(message):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    formatted_msg = f"[{timestamp}] {message}"
    with open(LOG_FILE, "a") as f:
        f.write(formatted_msg + "\n")
    print(formatted_msg, flush=True)

# ================= 2. Numba 原位极限优化内核 =================
# 彻底杜绝 Python 层的对象复制，单次 Worker 内存严格锁死在 10GB
@njit(fastmath=True, nogil=True)
def build_and_normalize(u_temp, k_opt, steps, n_bins, offset):
    # 使用 float32 砍掉一半内存，5w*5w float32 = 10GB
    counts = np.zeros((n_bins, n_bins), dtype=np.float32) 
    x = 0.5
    last_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
    
    warmup = 2000000 
    for i in range(warmup):
        L_i = np.log(i + offset)
        u_dyn = u_temp + k_opt / (L_i**2)
        x = 1.0 - u_dyn * x**2
        if x > 1.0: x = 0.999
        elif x < -1.0: x = -0.999
            
    for i in range(warmup, steps + warmup):
        L_i = np.log(i + offset)
        u_dyn = u_temp + k_opt / (L_i**2)
        x = 1.0 - u_dyn * x**2
        if x > 1.0: x = 0.999
        elif x < -1.0: x = -0.999
        
        current_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
        if 0 <= current_bin < n_bins and 0 <= last_bin < n_bins:
            counts[last_bin, current_bin] += 1.0 # float32 累加
        last_bin = current_bin
        
    # 🔥 C级别原位归一化，省去 P = counts/row_sums 的 10GB 内存拷贝
    for i in range(n_bins):
        row_sum = np.float32(0.0)
        for j in range(n_bins):
            row_sum += counts[i, j]
            
        if row_sum > 0.0:
            inv_sum = np.float32(1.0) / row_sum
            for j in range(n_bins):
                counts[i, j] *= inv_sum
        else:
            counts[i, i] = np.float32(1.0)
            
    return counts

# ================= 3. 特征值提取 (调用单精度求解器) =================
def extract_phases(P_normalized):
    # scipy.linalg.eigvals 遇到 float32 矩阵会自动路由到底层 sgeev，极大地节约内存
    vals = scipy.linalg.eigvals(P_normalized)
    valid = vals[vals.imag > 1e-5]
    return np.unwrap(np.sort(np.angle(valid)))

# ================= 4. 目标函数 =================
def objective_k(params, target_zeros, u_c, steps, n_bins, offset):
    k_opt = params[0]  
    t0 = time.time()
    
    t_end = 1.0 / (np.log(steps + offset)**2)
    u_temp = u_c - k_opt * t_end  
    
    try:
        # 1. Numba 内核执行（耗时约百秒级别）
        P_matrix = build_and_normalize(u_temp, k_opt, steps, n_bins, offset)
        t_matrix = time.time()
        
        # 2. 求解特征值（5w阶浮点对角化，视底层BLAS多线程效率，可能需要数小时）
        sys_phases = extract_phases(P_matrix)
        t_eig = time.time()
        
        N_actual = len(sys_phases)
        N_target = len(target_zeros)
        
        # 🛡️ 万阶拦截网：如果产生的特征值不够 10000 个，直接严厉惩罚
        N_comp = min(N_actual, N_target)
        if N_comp < 8000: 
            return 1e12 + (N_target - N_comp) * 1e6
            
        scale = target_zeros[0] / sys_phases[0]
        predicted = sys_phases[:N_comp] * scale
        
        mse = np.mean((predicted - target_zeros[:N_comp])**2)
        
        # 如果长度不够 10000，加上线性惩罚，迫使寻优收敛
        if N_comp < N_target:
            mse += (N_target - N_comp) * 500.0
            
        # 🔥 终极游击战防截断日志：k、scale、mse 全部原汁原味输出！方便随时停机！
        log_msg(f"[Worker] 尝试 k={k_opt!r} | 缩放={scale!r} | MSE={mse!r} | 提取阶数={N_actual} | 积分耗时={t_matrix-t0:.1f}s | 矩阵求解耗时={(t_eig-t_matrix)/3600:.3f}h")
        return mse
        
    except Exception as e: 
        log_msg(f"[-] Worker 崩溃 (可能为OOM): {e}")
        return 1e15

# ================= 5. 主程序 =================
if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)
    
    log_msg("="*80)
    log_msg("🚀 1D 宏观一万阶终极测绘启动 (N_BINS=50000, 500亿步)")
    log_msg("⚠️ 资源锁定配置: 16 Workers × 12 Threads = 192 Cores 联合作战")
    log_msg("="*80)

    # 加载事先算好的前 10,000 个黎曼零点
    try:
        target_zeros = np.load("riemann_10k_true.npy")[:10000]
        log_msg(f"[*] 成功加载靶向零点，总数: {len(target_zeros)}，最高能级: {target_zeros[-1]:.2f}")
    except Exception as e:
        log_msg(f"[-] 找不到或无法加载 riemann_10k_true.npy: {e}")
        exit()

    # --- 终极参数配置 ---
    U_C = 1.543689
    STEPS = 50_000_000_000  # 500 亿步绝热冷却
    OFFSET = 100000.0
    N_BINS = 50000          # 5w网格
    
    res = differential_evolution(
        func=objective_k, 
        bounds=[(5.0, 15.0)],
        args=(target_zeros, U_C, STEPS, N_BINS, OFFSET),
        strategy='best1bin', 
        maxiter=10, 
        popsize=128, 
        workers=16,         # ⚠️ 绝对不可调高！20是 480GB RAM 的极限生死线，16 是最安全的满载策略！
        updating='deferred',
        polish=False,
        disp=False
    )
    
    log_msg("\n" + "🏆"*20)
    log_msg(f"[*] 万阶圣杯已现！最优 k1 = {repr(res.x[0])}")
    log_msg(f"[*] 万阶极限 MSE = {repr(res.fun)}")
    log_msg("🏆"*20)