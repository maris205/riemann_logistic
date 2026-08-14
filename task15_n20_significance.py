"""
Task 15: Re-examine the N=20 residual "spike" without cherry-picking,
and run a proper significance test against the USTC ion-trap error bars.

Background: micro_ustc_data_match.ipynb ran the noisy ARPACK eigensolver
20 times and kept only the trial that MAXIMIZED |signed_diff[19]| (the N=20
residual) via score = err_sum_2_to_6 - 0.1*|n20_spike|. That is explicit
selection-on-the-outcome, so the reported N=20 "unexpectedly emerged" spike
is not a fair sample. This script:

  1. Rebuilds the exact same transition matrix (deterministic dynamics,
     no RNG in the map itself).
  2. Runs the eigendecomposition many times with different ARPACK restart
     vectors (the only source of run-to-run variability) and records the
     N=20 residual from EVERY trial without selecting on it.
  3. Reports the single-run (trial #1) result and the mean/std across
     trials, to see whether a spike near N=20 is a robust feature or an
     artifact of the original max-selection procedure.
  4. Performs an actual statistical test of the "coincidence" with the
     USTC error-bar pattern (Spearman rank correlation between our |residual|
     across N=1..20 and USTC's reported measurement uncertainty across
     N=1..20, plus a permutation test for whether N=20 specifically ranks
     unusually high in both series).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.stats import spearmanr
import mpmath
import time
import json
from pathlib import Path
from numba import njit

mpmath.mp.dps = 15

BEST_EPS = 0.001916
PLOT_N = 85
N_TRIALS = 20
COMPARE_N = 20

print("1. Computing true Riemann zeros...")
TRUE_ZEROS = np.array([float(mpmath.zetazero(i).imag) for i in range(1, PLOT_N + 1)])


@njit(fastmath=True, nogil=True)
def run_simulation(eps):
    steps = 10**6
    n_bins = 6000
    c_offset = 10.0
    mu_end = 1.5437
    delta_mu = 0.02
    t_start = 1.0 / (np.log(1 + c_offset) ** 2)
    t_end = 1.0 / (np.log(steps + c_offset) ** 2)
    k_opt = delta_mu / (t_start - t_end)
    u_c = mu_end - k_opt * t_end

    transitions = np.zeros((n_bins, n_bins), dtype=np.float64)
    V = np.zeros(n_bins, dtype=np.float64)
    dx = 2.0 / n_bins
    V[int(1.5 / dx)] = 1.0
    inv_2eps2 = 1.0 / (2.0 * eps ** 2)
    radius = int(5.0 * eps / dx) + 1

    for n in range(1, steps + 1):
        mu = u_c + k_opt / (np.log(n + c_offset) ** 2.0)
        mu = max(0.1, min(2.0, mu))
        V_next = np.zeros(n_bins, dtype=np.float64)
        for i in range(n_bins):
            if V[i] < 1e-12:
                continue
            x = -1.0 + dx * 0.5 + i * dx
            x_next = 1.0 - mu * x * x
            j_center = int((x_next + 1.0) / dx)
            j_start = max(0, j_center - radius)
            j_end = min(n_bins - 1, j_center + radius)
            w_sum = 0.0
            for j in range(j_start, j_end + 1):
                cj = -1.0 + dx * 0.5 + j * dx
                w_sum += np.exp(-(cj - x_next) ** 2 * inv_2eps2)
            if w_sum > 1e-18:
                inv_sum = 1.0 / w_sum
                for j in range(j_start, j_end + 1):
                    cj = -1.0 + dx * 0.5 + j * dx
                    prob = np.exp(-(cj - x_next) ** 2 * inv_2eps2) * inv_sum
                    flow = V[i] * prob
                    V_next[j] += flow
                    transitions[i, j] += flow
            else:
                jc = min(max(0, j_center), n_bins - 1)
                V_next[jc] += V[i]
                transitions[i, jc] += V[i]
        V = V_next
    return transitions


print(f"2. Building the transition matrix (eps={BEST_EPS}), one-time cost...")
t0 = time.time()
trans = run_simulation(BEST_EPS)
P_sparse = sp.csr_matrix(trans)
sums = np.array(P_sparse.sum(axis=1)).flatten()
sums[sums == 0] = 1.0
P_sparse.data /= sums[P_sparse.indices]
print(f"   Done in {time.time()-t0:.2f}s\n")

print(f"3. Running eigendecomposition {N_TRIALS} times WITHOUT selecting on N=20...")
all_signed_diffs = []
for trial in range(1, N_TRIALS + 1):
    vals, _ = eigs(P_sparse, k=450, which='LM', tol=1e-5)
    pos_vals = vals[vals.imag > 1e-4]
    phases = np.sort(np.angle(pos_vals))
    if len(phases) < PLOT_N:
        print(f"   [trial {trial:02d}] insufficient eigenvalues, skipped")
        continue
    pred_raw = phases[:PLOT_N]
    scale = TRUE_ZEROS[0] / pred_raw[0]
    pred_zeros = pred_raw * scale
    signed_diffs = pred_zeros - TRUE_ZEROS
    all_signed_diffs.append(signed_diffs)
    print(f"   [trial {trial:02d}] N=20 signed residual = {signed_diffs[19]:.4f}")

all_signed_diffs = np.array(all_signed_diffs)
n20_values = all_signed_diffs[:, 19]

print("\n" + "=" * 70)
print("UNSELECTED (no cherry-picking) results across all valid trials:")
print(f"  n_trials_valid = {len(n20_values)}")
print(f"  Trial #1 (single-run, first-in-sequence) N=20 residual = {n20_values[0]:.4f}")
print(f"  Mean N=20 residual across all trials = {n20_values.mean():.4f}")
print(f"  Std  N=20 residual across all trials  = {n20_values.std():.4f}")
print(f"  Min/Max N=20 residual across trials    = {n20_values.min():.4f} / {n20_values.max():.4f}")

# For reference: what the original cherry-picked notebook reported (selected to
# maximize |n20_spike| jointly with low err_sum_2_to_6). We recompute that
# selection criterion here on our own (unbiased) trial set for comparison only.
err_sum_2_to_6 = np.array([np.sum(np.abs(d[1:6])) for d in all_signed_diffs])
score = err_sum_2_to_6 - 0.1 * np.abs(n20_values)
best_idx = np.argmin(score)
print(f"\n  [For reference] Cherry-picked-style best trial index = {best_idx+1}, "
      f"N=20 residual if selected this way = {n20_values[best_idx]:.4f}")
print(f"  => Cherry-picked value vs. unselected mean: "
      f"{n20_values[best_idx]:.4f} vs {n20_values.mean():.4f} "
      f"(ratio = {abs(n20_values[best_idx])/ (abs(n20_values.mean())+1e-9):.2f}x)")
print("=" * 70 + "\n")

# Rank of |N=20 residual| among all N=1..PLOT_N residuals, per trial (unselected)
mean_abs_diffs = np.mean(np.abs(all_signed_diffs), axis=0)
rank_n20 = int(np.sum(mean_abs_diffs > mean_abs_diffs[19]) + 1)  # 1 = largest
print(f"Rank of |N=20 residual| among N=1..{PLOT_N} by mean |residual| across trials: "
      f"#{rank_n20} of {PLOT_N} (1=largest)")

# ================= 4. USTC comparison, using UNSELECTED (mean-of-trials) data =================
print("\n4. Statistical comparison against USTC ion-trap experimental uncertainty...")

raw_data_text = """
1,14.135,14.07(1),14.06(2),13.99(4),14.03(3)
2,21.022,21.04(2),21.00(2),20.93(5),20.82(3)
3,25.011,24.70(3),24.87(2),24.87(7),24.99(4)
4,30.425,30.59(2),30.31(2),30.29(3),30.27(4)
5,32.935,32.76(3),32.72(3),32.57(8),32.29(23)
6,37.586,37.64(2),37.62(2),37.39(2),37.59(4)
7,40.919,40.95(2),40.89(3),40.78(3),40.70(4)
8,43.327,42.85(9),43.12(4),43.23(4),42.74(40)
9,48.005,48.23(4),47.87(6),47.94(6),47.75(9)
10,49.774,49.26(19),49.67(3),49.36(23),49.23(22)
11,52.97,52.93(2),52.83(4),52.88(5),52.78(5)
12,56.446,56.56(3),56.58(3),56.28(3),56.49(5)
13,59.347,59.44(5),59.33(9),59.35(6),59.08(28)
14,60.832,60.10(48),60.13(414),60.41(144),60.67(9)
15,65.113,65.53(11),64.99(4),65.05(6),64.92(6)
16,67.08,67.06(5),67.10(3),66.98(10),66.50(40)
17,69.546,69.36(4),69.32(7),69.11(28),69.44(7)
18,72.067,71.82(3),71.84(3),71.76(8),71.95(7)
19,75.705,76.33(37),75.72(12),75.23(332),75.35(317)
20,77.145,76.84(8),77.41(6),76.80(9),76.82(83)
"""

import re


def parse_val_error(val_str):
    match = re.match(r"([\d\.]+)\((\d+)\)", val_str.strip())
    val_s = match.group(1)
    err_int = int(match.group(2))
    decimals = len(val_s.split('.')[1]) if '.' in val_s else 0
    err = err_int * (10 ** -decimals)
    return float(val_s), err


ustc_o16_err = []
for line in raw_data_text.strip().split('\n'):
    parts = line.split(',')
    _, o16_err = parse_val_error(parts[5])
    ustc_o16_err.append(o16_err)
ustc_o16_err = np.array(ustc_o16_err)  # N=1..20 reported uncertainty, Omega=16 MHz

our_mean_abs_residual_20 = mean_abs_diffs[:COMPARE_N]

rho, p_spearman = spearmanr(our_mean_abs_residual_20, ustc_o16_err)
print(f"  Spearman rank correlation (our unselected mean |residual| for N=1..20 "
      f"vs USTC Omega=16 MHz reported uncertainty): rho={rho:.4f}, p={p_spearman:.4f}")

# Permutation test: is N=20 unusually co-ranked as "large" in BOTH series,
# more than expected by chance under random relabeling of N?
rng = np.random.default_rng(42)
our_rank_n20 = int(np.sum(our_mean_abs_residual_20 > our_mean_abs_residual_20[19]) + 1)
ustc_rank_n20 = int(np.sum(ustc_o16_err > ustc_o16_err[19]) + 1)
joint_stat_observed = our_rank_n20 + ustc_rank_n20  # smaller = both rank N=20 as large
n_perm = 100_000
perm_stats = np.zeros(n_perm)
idx = np.arange(COMPARE_N)
for p in range(n_perm):
    perm = rng.permutation(idx)
    r_our = int(np.sum(our_mean_abs_residual_20 > our_mean_abs_residual_20[perm[19]]) + 1)
    r_ustc = int(np.sum(ustc_o16_err > ustc_o16_err[19]) + 1)
    perm_stats[p] = r_our + r_ustc
p_value_n20_joint = np.mean(perm_stats <= joint_stat_observed)

print(f"  Our unselected rank of N=20 by mean |residual| (1=largest): #{our_rank_n20} of {COMPARE_N}")
print(f"  USTC rank of N=20 by reported Omega=16 uncertainty (1=largest): #{ustc_rank_n20} of {COMPARE_N}")
print(f"  Permutation-test p-value for this joint-rank coincidence occurring by chance: "
      f"{p_value_n20_joint:.4f}")

results = {
    'n_trials_valid': int(len(n20_values)),
    'n20_residual_trial1_unselected': float(n20_values[0]),
    'n20_residual_mean_unselected': float(n20_values.mean()),
    'n20_residual_std_unselected': float(n20_values.std()),
    'n20_residual_min': float(n20_values.min()),
    'n20_residual_max': float(n20_values.max()),
    'n20_residual_cherrypicked_style': float(n20_values[best_idx]),
    'rank_n20_among_all_by_mean_abs_residual': rank_n20,
    'plot_n': PLOT_N,
    'spearman_rho_vs_ustc_o16_uncertainty': float(rho),
    'spearman_p_vs_ustc_o16_uncertainty': float(p_spearman),
    'our_rank_n20_within_1_20': our_rank_n20,
    'ustc_rank_n20_within_1_20': ustc_rank_n20,
    'permutation_p_value_joint_n20_coincidence': float(p_value_n20_joint),
}

output_dir = Path(__file__).resolve().parent / "task15_n20_significance"
output_dir.mkdir(exist_ok=True)
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_dir / 'results.json'}")
