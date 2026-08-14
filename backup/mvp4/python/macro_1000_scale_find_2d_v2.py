import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from numba import njit
from scipy.optimize import differential_evolution
import time
import mpmath
import multiprocessing

LOG_FILE = "macro_2d_v3.log"
def log_msg(message):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

@njit(fastmath=True, nogil=True)
def build_ulam_2d_kernel(u_c, k1, k2, steps, n_bins, offset):
    x = 0.5
    counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    last_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
    ln_end = np.log(steps + offset)
    u_temp = u_c - k1 / (ln_end**2) - k2 / (ln_end**3)
    warmup = 2000000 
    for i in range(warmup):
        ln_val = np.log(i + offset)
        u_dyn = u_temp + k1 / (ln_val**2) + k2 / (ln_val**3)
        x = 1.0 - u_dyn * x**2
        if x > 1.0: x = 0.999
        elif x < -1.0: x = -0.999
    for i in range(warmup, steps + warmup):
        ln_val = np.log(i + offset)
        u_dyn = u_temp + k1 / (ln_val**2) + k2 / (ln_val**3)
        x = 1.0 - u_dyn * x**2
        if x > 1.0: x = 0.999
        elif x < -1.0: x = -0.999
        curr_bin = int((x + 1.0) / 2.0 * (n_bins - 1))
        if 0 <= curr_bin < n_bins and 0 <= last_bin < n_bins:
            counts[last_bin, curr_bin] += 1
        last_bin = curr_bin
    return counts

def extract_phases(counts):
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = counts / row_sums
    vals = np.linalg.eigvals(P)
    valid = vals[vals.imag > 1e-5]
    return np.unwrap(np.sort(np.angle(valid)))

def objective_2d(params, target_zeros, u_c, steps, n_bins, offset):
    k1, k2 = params
    t0 = time.time()
    try:
        counts = build_ulam_2d_kernel(u_c, k1, k2, steps, n_bins, offset)
        sys_phases = extract_phases(counts)
        N_actual = len(sys_phases)
        if N_actual < 800: return 1e12 + (1000 - N_actual) * 1e6
        scale = target_zeros[0] / sys_phases[0]
        predicted = sys_phases * scale
        mse = np.mean((predicted[:1000] - target_zeros[:1000])**2)
        log_msg(f"[Worker] k1={k1!r}, k2={k2!r} | MSE={mse:.6f} | N={N_actual} | {time.time()-t0:.1f}s")
        return mse
    except: return 1e15

if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)
    mpmath.mp.dps = 25
    zeros = np.array([float(mpmath.zetazero(i).imag) for i in range(1, 1001)])
    log_msg("🚀 2D 终极万维版启动 (N_BINS=10000)")
    res = differential_evolution(
        func=objective_2d, bounds=[(5.0, 15.0), (-5.0, 35.0)],
        args=(zeros, 1.543689, 10_000_000_000, 10000, 100000.0),
        strategy='best1bin', maxiter=10, popsize=56, workers=112, polish=False
    )
    log_msg(f"🏆 2D 圣杯: k1={res.x[0]!r}, k2={res.x[1]!r} | MSE={res.fun!r}")