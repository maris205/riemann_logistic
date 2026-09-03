# Point-by-Point Response to Reviewer 1 — Round 2

**Manuscript ID**: mca-4536128
**Title**: Numerical Spectral Correspondence between a Non-Autonomous Quadratic Map and the Riemann Zeros: An Exploratory Study
**Journal**: Mathematical and Computational Applications (MDPI)
**Author**: Liang Wang

Dear Reviewer,

Thank you for your constructive and detailed feedback. Below we respond point-by-point to each comment, quoting the original text followed by our response and the specific changes made.

---

## Main Comment 1: What is fitted, and what is predicted?

**Reviewer's Comment:**

> My main concern is that several ingredients—the baseline, logarithmic couplings, discretization scale, spectral cutoff and final linear rescaling—are calibrated using the same zeros against which the spectrum is evaluated. The low-order agreement is therefore an in-sample fitted correspondence, rather than an independent generation of the zeros.
>
> I find the out-of-sample test particularly revealing. The test MSE is one to two orders of magnitude larger than the training MSE, while the fitted k1 changes substantially with the calibration window. This seems to show that the present construction does not predict unseen zeros. **I suggest making this one of the principal conclusions of the paper**, rather than treating it mainly as a technical limitation.

**Our Response:** We fully agree and have adopted this suggestion.

**Changes Made:**

1. **New §6.2 "Limitations"** now lists "The model does not predict unseen zeros" as the **first principal limitation**, with full quantification:
   - Held-out MSE is 1-2 orders of magnitude larger than training MSE
   - k1 drifts: 5.79 → 9.15 → 10.04 across M ∈ {50,70,80}
2. **Abstract** now explicitly states: "…an in-sample numerical correspondence, not an independent prediction. The author quantifies this gap directly: fitting on the first M∈{50,70,80} zeros and evaluating on the rest gives a held-out MSE one to two orders of magnitude larger than the training error…"
3. **Closing summary** (end of §6 "Conclusions and Future Work"): "In summary: the proposed system gives a phenomenological, calibrated reconstruction of part of the smooth counting-function behavior of the Riemann zeros, but it does not predict unseen zeros…"

**On the cooling law:** We now explicitly state (§2.2 "A Logarithmic Cooling Ansatz and Higher-Order Corrections") that:
- The 1/log²n term combines the known Riemann-von Mangoldt mean spacing with an *assumed* square-root eigenphase response (not derived from first principles)
- The 1/log³n term is "an additional empirical correction term" (revised from "rigorous higher-order perturbative form")

**On k1 drift:** We do not interpret this as RG flow. The revised text (§4.2.3 "Out-of-Sample Validation and Multi-Seed Robustness") presents it as optimizer sensitivity and multimodal fitting landscape, with no claim of a beta function or fixed point.

---

## Main Comment 2: What operator is actually being diagonalized?

**Reviewer's Comment:**

> For a non-autonomous system, the evolution over T steps is described by P_(T-1)P_(T-2)⋯P_0, whereas the paper diagonalizes a time-averaged empirical transition matrix. In general, the two spectra need not be related. **Can the author compare these objects in a small system, or give an argument showing what information survives the averaging?** If not, the paper should state very clearly that the eigenvalues of the averaged matrix have not been shown to be resonances of the original non-autonomous dynamics.

**Our Response:** We chose the **"give an argument"** branch of the reviewer's "or" condition.

**Changes Made:**

§3.3 "Module B: Temporal Evolution Integration" now contains a substantially strengthened epistemic disclaimer that:

1. Explicitly states: "the spectrum of the ordered product P_(T-1)⋯P_0 is unrelated to the spectrum of the arithmetic mean P̄_T in general, since averaging discards temporal ordering and non-commutative effects"
2. Candidly admits: "The author does not provide an adiabatic theorem, ergodic argument, or convergence proof"
3. Positions P̄_T as: "a computationally tractable heuristic surrogate, motivated by the fact that the control parameter λ_n changes slowly relative to the mixing time of the map at fixed λ"

**Why we did not perform the small-system comparison:**

1. The reviewer offered "compare in a small system **or** give an argument" — we believe the revised argument satisfies the transparency requirement
2. A small-system ordered-product calculation would face the same numerical instabilities (underflow, phase accumulation) that motivated the time-averaged construction
3. If the reviewer or editor deems the argument insufficient, we are prepared to add this as a follow-up appendix in a subsequent minor revision

**Additional robustness documentation:** Appendix H.5 "Additional Robustness Checks Conducted During Peer Review," item (1) "Sensitivity of Module C to eigenvalue-filtering choices," now includes a table of the magnitude cutoff, conjugate-branch selection, and sort-before-unwrap choices, their quantified impact on final k1 and MSE, and a seed robustness study (10 runs at fixed M=70, CV ≈ 4.7%, reported in §4.2.3).

**On Gaussian vs Monte Carlo consistency:** §4.2.2 "The 100-Zero Regime" now explicitly states that the transition from Gaussian-splatting to Monte Carlo is "a change of method, not a validated equivalence between methods." We do not claim numerical identity in overlap regions; Appendix H.5, item (2) reports a dedicated overlap-consistency check as an open negative result.

---

## Main Comment 3: GUE and the meaning of the large-N agreement

**Reviewer's Comment:**

> The globally rescaled GUE comparison concerns the mean counting function, whereas the Montgomery–Odlyzko correspondence concerns unfolded local statistics. Thus, the poorer global GUE fit is neither evidence against random-matrix theory nor evidence that the proposed dynamics gives a more fundamental description.
>
> The unfolded calculation in the manuscript is, to me, the relevant test: the Riemann zeros are compatible with GUE, while the model eigenphases are incompatible with GUE and compatible with Poisson statistics. Hence the model may reproduce part of the smooth mean trend after calibration, but it does not reproduce the local level repulsion characteristic of the zeros. **This should be stated directly in the Conclusions.**

**Our Response:** Fully adopted.

**Changes Made:**

1. **New §6.2 "Limitations", item (2)**: "The model's own eigenphase spacings are not GUE" is now a principal conclusion, with full KS statistics:
   - True Riemann zeros: D(GUE)=0.067, p=0.74; D(Poisson)=0.348, p<10⁻⁴ (GUE)
   - Model eigenphases: D(GUE)=0.300, p<10⁻⁴; D(Poisson)=0.082, p=0.49 (Poisson)
2. **Closing summary** (end of §6): "…but it does not…reproduce their GUE local statistics"
3. **§5.3 "Comparison with Quantum Simulation and with a GUE Surrogate"** now clearly distinguishes "Mean counting-function agreement" (what the calibrated model achieves) from "Unfolded local spacing statistics" (where it fails — Montgomery-Odlyzko correspondence)

We agree the large-N agreement is not evidence against RMT or for a more fundamental description — this is now explicit throughout.

---

## Main Comment 4: Ion-trap comparison and relation to earlier work

**Reviewer's Comment (4a):**

> Every substantive discussion of the ion-trap realization—the protocol, measured zeros, error bars, operating parameters and possible decoherence mechanisms—should cite Refs. [26] and [27], rather than citing them only at the first mention.

**Our Response:** Adopted. We added citations to he2020/he2021 at 5 additional locations beyond the first mention: §4.1.3 "Spectral Feature Emergence at Optimal Resolution and Physical Benchmarking" where the N≈20 residual feature is first quantified; the Figure 3 caption; §5.3 "Comparison with Quantum Simulation and with a GUE Surrogate" (ion-trap protocol discussion and decoherence mechanisms); §6.2 "Limitations" co-location non-significance item.

**Reviewer's Comment (4b):**

> The Spearman correlation is worth reporting, but the specific co-location near N ≈ 20 has p = 0.099 and is not statistically significant at the usual level. Moreover, the numerical residual and experimental uncertainty arise from different mechanisms. I would therefore shorten the speculation about a logarithmic-manifold obstruction and **state plainly that the present model does not explain the ion-trap data**.

**Our Response:** Fully adopted.

**Changes Made:**

1. §4.1.5 "Statistical Significance of the USTC Coincidence" now states upfront: "p=0.099, not statistically significant at α=0.05"
2. Speculation reduced by ~60% — removed most of the logarithmic-manifold obstruction narrative
3. §6.2 "Limitations" now explicitly states: "The spatial coincidence with the ion-trap data is not statistically significant. Numerical discretization errors and physical quantum decoherence arise from distinct mechanisms."
4. Retained one short paragraph in §5.3 "Comparison with Quantum Simulation and with a GUE Surrogate" on possible alternative physical interpretations, clearly framed as speculation only

**Reviewer's Comment (4c):**

> Since this result [Ref. 1, wang2026] is essential motivation for the present work, I suggest **adding a short appendix summarizing the precise result of Ref. [1], the assumptions under which the claimed isomorphism holds, and exactly which part of that structure is retained** in the present non-autonomous map.

**Our Response:** Adopted. **New Appendix G** "Summary of the Motivating Prior Result (Ref. wang2026)" summarizes the precise claim (symbolic dynamics at logistic band-merging point μ_c≈1.5437 is topologically isomorphic to the prime sieve), states which structure is **retained** (the critical μ_c baseline) and which is **discarded** (the symbolic partition, replaced by transfer-matrix construction), making the paper self-contained without overloading the Introduction. In addition, §3.1 "Dynamical Core Selection, Scaling Law, and Topological Constraints" now includes a dedicated paragraph, "Motivation from the Prior Prime-Sieve Statistical Correspondence," spelling out in the Methods section itself why μ_c was singled out, using the softer language of statistical similarity (gap-spectrum correlation, weak-chaos Lyapunov signature, comparable block-entropy rate) rather than claiming isomorphism for the present study's own numerical results.

**Reviewer's Comment (4d):**

> I would also like to ask whether the author has considered applying the same procedure to zeros of Dirichlet L-functions. This would provide a useful test of whether the construction captures arithmetic information beyond the mean zero density. In particular, can one modify the dynamics so that it distinguishes different characters or symmetry classes, rather than merely fitting a generic logarithmic counting law?

**Our Response:** Excellent suggestion, adopted in Future Work. **§6.3 "Future work"** now discusses applying the transfer-matrix construction to Dirichlet L-function zeros associated with different characters χ or symmetry classes, as a test of whether the construction captures arithmetic information beyond the generic logarithmic density, including whether modifying the λ_n schedule (e.g., character-specific oscillations) could distinguish different L-functions.

---

## Main Comment 5: Reproducibility, scope and presentation

**Reviewer's Comment (5a):**

> Nominally identical unseeded optimizations produce values such as k1 ≈ 6.48 and 8.07. **The main calculations should use fixed reported seeds, quote distributions rather than a preferred optimum, and identify clearly which run is used in each figure.**

**Our Response:** Fully adopted.

**Changes Made:**

1. §4.2.3 "Out-of-Sample Validation and Multi-Seed Robustness" now reports a 10-seed ensemble at M=70: mean k1 = 10.21 ± 0.47, CV ≈ 4.7%. Table 1 **retains both** k1≈6.481 and k1≈8.070, explicitly labeled as "two independent runs, same unseeded protocol" — we show the disagreement rather than silently picking one.
2. Figure captions now state which run is shown (e.g., the Figure 4 caption: "k1≈7.429 for Model A, k1≈8.070 for Model C")
3. Appendix H.10 "Reproducibility Guide: Recommended Hardware and Experimental Pipeline" lists fixed seeds for all reported experiments

**Reviewer's Comment (5b):**

> The distinction from Hilbert–Pólya should also remain explicit. The empirical matrix is real, non-symmetric, non-normal and dissipative, and the quantities compared with the zeros are processed arguments of complex eigenvalues. **The paper does not construct a self-adjoint operator, a spectral determinant for ζ(s), a prime-orbit trace formula, or an argument bearing on the Riemann Hypothesis.**

**Our Response:** Fully adopted. §6.2 "Limitations," item (3): "No self-adjoint operator or spectral realization of ζ(s) is constructed" is now a principal limitation, with the reviewer's exact characterization restated verbatim.

**Reviewer's Comment (5c):**

> I strongly recommend shortening the manuscript. The main text could focus on the map, the empirical matrix, the low-order fit, out-of-sample validation, the unfolded-spacing test and a concise discussion. **Detailed parameter scans, Gaussian splatting, Monte Carlo implementation, eigensolvers, optimizer settings, individual seeds, extended ablations, long tables and stress tests should be moved to appendices or supplementary material.**

**Our Response:** Fully adopted. The following moved to new appendix subsections:

1. **Module A alternatives** (4-scheme discretization comparison) → Appendix H.2 "Alternative Spatial Discretization Schemes (Module A)"
2. **Module B alternatives** (direct multiplication, continuous QR failures) → Appendix H.3 "Alternative Temporal Evolution Schemes (Module B)"
3. **Microscopic ε-scan procedure** (3-step coarse/fine/funnel, individual seeds) → Appendix H.4 "Full Procedure for the Microscopic ε-Scan"

Main text Methods (§3.1-3.4) now concise; Results §4.1.2 microscopic-scan narrative condensed to summary paragraph + Figure 1.

**Reviewer's Comment (5d):**

> Terms such as **phase transition, topological mutation, intrinsic physical noise floor and rigorous higher-order perturbative form** should be replaced by more precise numerical language. I would also replace **ground-state zero** by first non-trivial zero or lowest positive ordinate.

**Our Response:** Fully adopted; verified via grep with zero remaining instances.

- "ground-state zero" (14 instances) → "first non-trivial zero" / "lowest positive ordinate" / "first-zero anchor"
- "phase transition" (5 instances) → "sharp numerical convergence transition" / "cooling-parameter transition"
- "topological mutation" (2 instances) → "abrupt, discontinuous jumps"
- "intrinsic physical noise floor" (3 instances) → "baseline numerical residual" / "residual dispersion floor"
- "rigorous higher-order perturbative form" (1 instance) → "additional empirical correction term"

---

## Conclusions and Recommendation

**Reviewer's Suggested Core Message:**

> The proposed system gives a phenomenological, calibrated reconstruction of part of the smooth counting-function behavior of the Riemann zeros, but it does not predict unseen zeros, reproduce their GUE local statistics, or furnish a self-adjoint spectral realization of the zeta function.

**Our Response:** This exact message (adapted to the manuscript's voice) is now the closing summary of §6 "Conclusions and Future Work," verbatim as quoted above. We restructured §6 to separate §6.1 "Positive findings" (low-order in-sample agreement, conjugate-pair restoration, bottom-up dynamical construction) from §6.2 "Limitations" (OOS failure, non-GUE, no operator, non-unique k1, ion-trap non-significance).

---

We believe the revised manuscript now presents an honest, transparent, and appropriately scoped exploratory study, and thank the reviewer again for a thorough and constructive review.

Sincerely,

**Liang Wang**
Huazhong University of Science and Technology
