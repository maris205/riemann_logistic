import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import time
import os

# --- 1. 配置独立目录 ---
SAVE_DIR = "autonomous_limit_sigma"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f">>> 已创建独立统计目录: {SAVE_DIR}")

# --- 2. JIT 极致演化内核 (k=0) ---
@njit(fastmath=True, nogil=True)
def compute_ultra_autonomous(u_c, steps, n_bins):
    x = 0.5
    # 深度热启动
    for i in range(1000000):
        x = 1 - u_c * x**2
    
    counts = np.zeros((n_bins, n_bins), dtype=np.float64) 
    last_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
    
    # 暴力万亿步：彻底平滑相空间测度
    for i in range(steps):
        x = 1 - u_c * x**2
        current_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
        if 0 <= current_bin < n_bins:
            counts[last_bin, current_bin] += 1
            last_bin = current_bin
    return counts

def run_limit_experiment():
    U_C = 1.543689012  
    N_BINS = 20000     # 锁定黄金分辨率
    STEPS = 10**11     # 万亿步极限压制
    TARGET_K = 1000    # 索要 1000 个特征值进行大样本统计
    
    print(f">>> 启动万亿步自治极限测试 | 目标: 0.422 (GUE)")
    t0 = time.time()
    
    # 执行演化
    counts = compute_ultra_autonomous(U_C, STEPS, N_BINS)
    
    # 算子构建与求解
    P = csr_matrix(counts)
    row_sums = np.array(P.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    P = P.multiply(1.0 / row_sums[:, np.newaxis])
    
    print(f">>> 正在提取 {TARGET_K} 阶特征谱...")
    vals, _ = eigs(P, k=TARGET_K, which='LM', tol=1e-8)
    
    # 统计计算
    phases = np.sort(np.angle(vals[np.abs(vals) > 0.4]))
    s = np.diff(phases) / np.mean(np.diff(phases))
    sigma = np.std(s)
    
    # 保存结果到独立目录 [cite: 2026-02-10]
    np.save(os.path.join(SAVE_DIR, f"phases_limit_sigma_{sigma:.4f}.npy"), phases)
    
    # 自动出图：GUE 拟合直方图
    plt.figure(figsize=(10, 6))
    plt.hist(s, bins=80, density=True, alpha=0.6, color='#3498db', label=f'Model ($\sigma$={sigma:.4f})')
    
    s_theory = np.linspace(0, 4, 100)
    p_gue = (32 / np.pi**2) * (s_theory**2) * np.exp(-4 * s_theory**2 / np.pi)
    plt.plot(s_theory, p_gue, 'r-', lw=3, label='Target GUE (0.422)')
    
    plt.title(f'Figure 1: Autonomous Limit Statistics (Steps=10^11)', fontsize=14)
    plt.xlabel('Normalized Spacing s')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'GUE_Limit_Verification.png'), dpi=300)
    
    print(f"\n实验完成！结果已存入 {SAVE_DIR} 目录。")
    print(f"最终 Sigma: {sigma:.4f}")

if __name__ == "__main__":
    run_limit_experiment()