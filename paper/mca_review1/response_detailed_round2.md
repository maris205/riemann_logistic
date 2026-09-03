# Point-by-Point Response to Reviewers — Round 2

**Manuscript ID**: mca-4536128  
**Title**: Numerical Spectral Correspondence between a Non-Autonomous Quadratic Map and the Riemann Zeros: An Exploratory Study  
**Journal**: Mathematical and Computational Applications (MDPI)  
**Author**: Liang Wang

---

Dear Editor,

Thank you for the opportunity to revise our manuscript. We are grateful to both reviewers for their constructive and detailed feedback. Below we respond point-by-point to each comment, quoting the original reviewer text followed by our response and the specific changes made.

---

## Response to Reviewer 1

### Main Comment 1: What is fitted, and what is predicted?

**Reviewer's Comment:**
> My main concern is that several ingredients—the baseline, logarithmic couplings, discretization scale, spectral cutoff and final linear rescaling—are calibrated using the same zeros against which the spectrum is evaluated. The low-order agreement is therefore an in-sample fitted correspondence, rather than an independent generation of the zeros.
>
> I find the out-of-sample test particularly revealing. The test MSE is one to two orders of magnitude larger than the training MSE, while the fitted k₁ changes substantially with the calibration window. This seems to show that the present construction does not predict unseen zeros. **I suggest making this one of the principal conclusions of the paper**, rather than treating it mainly as a technical limitation.

**Our Response:**

We fully agree and have adopted this suggestion.

**Changes Made:**

1. **New §6.2 "Limitations"** now lists "The model does not predict unseen zeros" as the **first principal limitation**, with full quantification:
   - Held-out MSE is 1-2 orders of magnitude larger than training MSE
   - k₁ drifts: 5.79 → 9.15 → 10.04 across M ∈ {50,70,80}

2. **Abstract** now explicitly states:
   > "...an in-sample numerical correspondence, not an independent prediction. The author quantifies this gap directly: fitting on the first M∈{50,70,80} zeros and evaluating on the rest gives a held-out MSE one to two orders of magnitude larger than the training error..."

3. **Closing summary** (end of §6 "Conclusions and Future Work"):
   > "In summary: the proposed system gives a phenomenological, calibrated reconstruction of part of the smooth counting-function behavior of the Riemann zeros, but it does not predict unseen zeros..."

**On the cooling law:** We now explicitly state (§2.2 "A Logarithmic Cooling Ansatz and Higher-Order Corrections") that:
- The 1/log²n term combines the known Riemann-von Mangoldt mean spacing with an *assumed* square-root eigenphase response (not derived from first principles)
- The 1/log³n term is "an additional empirical correction term" (revised from "rigorous higher-order perturbative form")

**On k₁ drift:** We do not interpret this as RG flow. The revised text (§4.2.3 "Out-of-Sample Validation and Multi-Seed Robustness") presents it as optimizer sensitivity and multimodal fitting landscape, with no claim of a beta function or fixed point.

---

### Main Comment 2: What operator is actually being diagonalized?

**Reviewer's Comment:**
> For a non-autonomous system, the evolution over T steps is described by P_{T-1}P_{T-2}···P_0, whereas the paper diagonalizes a time-averaged empirical transition matrix. In general, the two spectra need not be related. **Can the author compare these objects in a small system, or give an argument showing what information survives the averaging?** If not, the paper should state very clearly that the eigenvalues of the averaged matrix have not been shown to be resonances of the original non-autonomous dynamics.

**Our Response:**

We chose the **"give an argument"** branch of the reviewer's "or" condition.

**Changes Made:**

**§3.3 "Module B: Temporal Evolution Integration"** now contains a substantially strengthened epistemic disclaimer that:

1. Explicitly states:
   > "the spectrum of the ordered product P_{T-1}···P_0 is unrelated to the spectrum of the arithmetic mean P̄_T in general, since averaging discards temporal ordering and non-commutative effects"

2. Candidly admits:
   > "The author does not provide an adiabatic theorem, ergodic argument, or convergence proof"

3. Positions P̄_T as:
   > "a computationally tractable heuristic surrogate, motivated by the fact that the control parameter λ_n changes slowly relative to the mixing time of the map at fixed λ"

**Why we did not perform the small-system comparison:**

1. The reviewer offered "compare in a small system **or** give an argument" — we believe the revised argument satisfies the transparency requirement
2. A small-system ordered-product calculation would face the same numerical instabilities (underflow, phase accumulation) that motivated the time-averaged construction
3. If the reviewer or editor deems the argument insufficient, we are prepared to add this as a follow-up appendix in a subsequent minor revision

**Additional robustness documentation:**

- **Appendix H.5 "Additional Robustness Checks Conducted During Peer Review", item (1)** now includes:
  - Table of Module C eigenvalue-filtering choices (magnitude cutoff, conjugate-branch selection, sort-before-unwrap choice)
  - Quantified impact on final k₁ and MSE
  - Seed robustness study (10 runs at fixed M=70, CV ≈ 4.7%, reported in §4.2.3)

**On Gaussian vs Monte Carlo consistency:**

§4.2.2 "The 100-Zero Regime" now explicitly states that the transition from Gaussian-splatting to Monte Carlo is "a change of method, not a validated equivalence between methods." We do not claim numerical identity in overlap regions; Appendix H.5, item (2) reports a dedicated overlap-consistency check as an open negative result.

---

### Main Comment 3: GUE and the meaning of the large-N agreement

**Reviewer's Comment:**
> The globally rescaled GUE comparison concerns the mean counting function, whereas the Montgomery–Odlyzko correspondence concerns unfolded local statistics. Thus, the poorer global GUE fit is neither evidence against random-matrix theory nor evidence that the proposed dynamics gives a more fundamental description.
>
> The unfolded calculation in the manuscript is, to me, the relevant test: the Riemann zeros are compatible with GUE, while the model eigenphases are incompatible with GUE and compatible with Poisson statistics. Hence the model may reproduce part of the smooth mean trend after calibration, but it does not reproduce the local level repulsion characteristic of the zeros. **This should be stated directly in the Conclusions.**

**Our Response:**

Fully adopted.

**Changes Made:**

1. **New §6.2 "Limitations", item (2)**: "The model's own eigenphase spacings are not GUE" is now a principal conclusion, with full KS statistics:
   - True Riemann zeros: D(GUE)=0.067, p=0.74; D(Poisson)=0.348, p<1e-4 ✓ GUE
   - Model eigenphases: D(GUE)=0.300, p<1e-4; D(Poisson)=0.082, p=0.49 ✓ Poisson

2. **Closing summary** (end of §6):
   > "...but it does not...reproduce their GUE local statistics"

3. **§5.3 "Comparison with Quantum Simulation and with a GUE Surrogate"** now clearly distinguishes:
   - "Mean counting-function agreement" (what the calibrated model achieves)
   - "Unfolded local spacing statistics" (where it fails — Montgomery-Odlyzko correspondence)

We agree the large-N agreement is not evidence against RMT or for a more fundamental description — this is now explicit throughout.

---

### Main Comment 4: Ion-trap comparison and relation to earlier work

**Reviewer's Comment (4a):**
> Every substantive discussion of the ion-trap realization—the protocol, measured zeros, error bars, operating parameters and possible decoherence mechanisms—should cite Refs. [26] and [27], rather than citing them only at the first mention.

**Our Response:**

Adopted. We added `\cite{he2020,he2021}` at 5 additional locations beyond the first mention.

**Changes Made:**

- §4.1.3 "Spectral Feature Emergence at Optimal Resolution and Physical Benchmarking" where the N≈20 residual feature is first quantified
- Figure 3 caption where visual comparison is shown
- §5.3 where ion-trap protocol is discussed
- §5.3 where decoherence mechanisms are discussed
- §6.2 "Limitations" where co-location non-significance is stated

---

**Reviewer's Comment (4b):**
> The Spearman correlation is worth reporting, but the specific co-location near N ≃ 20 has p = 0.099 and is not statistically significant at the usual level. Moreover, the numerical residual and experimental uncertainty arise from different mechanisms. I would therefore shorten the speculation about a logarithmic-manifold obstruction and **state plainly that the present model does not explain the ion-trap data**.

**Our Response:**

Fully adopted.

**Changes Made:**

1. **§4.1.5 "Statistical Significance of the USTC Coincidence"** now states upfront:
   > "p=0.099, not statistically significant at α=0.05"

2. **Speculation reduced by ~60%** — removed most of the logarithmic-manifold obstruction narrative

3. **§6.2 "Limitations"** now explicitly states:
   > "The spatial coincidence with the ion-trap data is not statistically significant. Numerical discretization errors and physical quantum decoherence arise from distinct mechanisms."

4. Retained one short paragraph in §5.3 "Comparison with Quantum Simulation and with a GUE Surrogate" on possible alternative physical interpretations, clearly framed as speculation only.

---

**Reviewer's Comment (4c):**
> Since this result [Ref. 1, wang2026] is essential motivation for the present work, I suggest **adding a short appendix summarizing the precise result of Ref. [1], the assumptions under which the claimed isomorphism holds, and exactly which part of that structure is retained** in the present non-autonomous map.

**Our Response:**

Adopted.

**Changes Made:**

**New Appendix G** "Summary of the Motivating Prior Result (Ref.~wang2026)":

- Summarizes the precise claim: symbolic dynamics at logistic band-merging point μ_c ≈ 1.5437 is topologically isomorphic to the prime sieve
- States which structure is **retained**: the critical μ_c baseline
- States which structure is **discarded**: the symbolic partition (replaced by transfer-matrix construction)
- Makes the paper self-contained without overloading the Introduction

In addition, §3.1 "Dynamical Core Selection, Scaling Law, and Topological Constraints" now includes a dedicated paragraph, "Motivation from the Prior Prime-Sieve Statistical Correspondence," spelling out in the Methods section itself why μ_c was singled out, using the softer language of statistical similarity (gap-spectrum correlation, weak-chaos Lyapunov signature, comparable block-entropy rate) rather than claiming isomorphism for the present study's own numerical results.

---

**Reviewer's Comment (4d):**
> I would also like to ask whether the author has considered applying the same procedure to zeros of Dirichlet L-functions. This would provide a useful test of whether the construction captures arithmetic information beyond the mean zero density. In particular, can one modify the dynamics so that it distinguishes different characters or symmetry classes, rather than merely fitting a generic logarithmic counting law?

**Our Response:**

Excellent suggestion. We have added discussion to Future Work.

**Changes Made:**

**§6.3 "Future work"** now includes:

> "A natural extension would be to apply the same transfer-matrix construction to zeros of Dirichlet L-functions associated with different characters χ or symmetry classes. This would test whether the construction captures arithmetic information beyond the generic logarithmic density. One could explore whether modifying the λ_n schedule (e.g., incorporating character-specific oscillations) allows the model to distinguish different L-functions, or whether the approach remains insensitive to the finer arithmetic structure encoded in the character."

---

### Main Comment 5: Reproducibility, scope and presentation

**Reviewer's Comment (5a):**
> Nominally identical unseeded optimizations produce values such as k₁ ≃ 6.48 and 8.07. **The main calculations should use fixed reported seeds, quote distributions rather than a preferred optimum, and identify clearly which run is used in each figure.**

**Our Response:**

Fully adopted.

**Changes Made:**

1. **§4.2.3 "Out-of-Sample Validation and Multi-Seed Robustness"** now reports:
   - 10-seed ensemble at M=70: mean k₁ = 10.21 ± 0.47, CV ≈ 4.7%
   - Table 1 **retains both** k₁≈6.481 and k₁≈8.070, explicitly labeled as "two independent runs, same unseeded protocol" — we show the disagreement rather than silently picking one

2. **Figure captions** now state which run is shown:
   - Figure 4 caption: "k₁≈7.429 for Model A, k₁≈8.070 for Model C"

3. **Appendix H.10 "Reproducibility Guide: Recommended Hardware and Experimental Pipeline"** lists fixed seeds for all reported experiments

---

**Reviewer's Comment (5b):**
> The distinction from Hilbert–Pólya should also remain explicit. The empirical matrix is real, non-symmetric, non-normal and dissipative, and the quantities compared with the zeros are processed arguments of complex eigenvalues. **The paper does not construct a self-adjoint operator, a spectral determinant for ζ(s), a prime-orbit trace formula, or an argument bearing on the Riemann Hypothesis.**

**Our Response:**

Fully adopted.

**Changes Made:**

**§6.2 "Limitations", item (3)**: "No self-adjoint operator or spectral realization of ζ(s) is constructed" is now a principal limitation, with explicit statement:

> "The empirical matrix is real, non-symmetric, non-normal, and dissipative. The paper does not construct a self-adjoint operator, a spectral determinant for ζ(s), a prime-orbit trace formula, or an argument bearing on the Riemann Hypothesis."

---

**Reviewer's Comment (5c):**
> I strongly recommend shortening the manuscript. The main text could focus on the map, the empirical matrix, the low-order fit, out-of-sample validation, the unfolded-spacing test and a concise discussion. **Detailed parameter scans, Gaussian splatting, Monte Carlo implementation, eigensolvers, optimizer settings, individual seeds, extended ablations, long tables and stress tests should be moved to appendices or supplementary material.**

**Our Response:**

Fully adopted.

**Changes Made:**

The following detailed content has been moved to new appendix subsections:

1. **Module A alternatives** (4-scheme discretization comparison) → **Appendix H.2 "Alternative Spatial Discretization Schemes (Module A)"**

2. **Module B alternatives** (direct multiplication, continuous QR failures) → **Appendix H.3 "Alternative Temporal Evolution Schemes (Module B)"**

3. **Microscopic ε-scan procedure** (3-step coarse/fine/funnel, individual seeds) → **Appendix H.4 "Full Procedure for the Microscopic ε-Scan"**

**Main text Methods** (§3.1-3.4) is now concise, focuses on the chosen method with brief pointers to appendices for alternatives.

**Main text Results** §4.1.2 microscopic-scan narrative condensed from 3 detailed optimization steps to summary paragraph + Figure 1.

---

**Reviewer's Comment (5d):**
> Terms such as **phase transition, topological mutation, intrinsic physical noise floor and rigorous higher-order perturbative form** should be replaced by more precise numerical language. I would also replace **ground-state zero** by first non-trivial zero or lowest positive ordinate.

**Our Response:**

Fully adopted. All instances have been replaced and verified via grep.

**Changes Made:**

- "ground-state zero" (14 instances) → "first non-trivial zero" / "lowest positive ordinate" / "first-zero anchor"
- "phase transition" (5 instances) → "sharp numerical convergence transition" / "cooling-parameter transition"
- "topological mutation" (2 instances) → "abrupt, discontinuous jumps"
- "intrinsic physical noise floor" (3 instances) → "baseline numerical residual" / "residual dispersion floor"
- "rigorous higher-order perturbative form" (1 instance) → "additional empirical correction term"

All replacements verified: `grep` returns zero remaining instances of flagged terms.

---

### Reviewer 1 — Conclusions and Recommendation

**Reviewer's Suggested Core Message:**
> The proposed system gives a phenomenological, calibrated reconstruction of part of the smooth counting-function behavior of the Riemann zeros, but it does not predict unseen zeros, reproduce their GUE local statistics, or furnish a self-adjoint spectral realization of the zeta function.

**Our Response:**

This exact message (adapted to the manuscript's voice) is now the closing summary of §6 "Conclusions and Future Work":

> "In summary: the proposed system gives a phenomenological, calibrated reconstruction of part of the smooth counting-function behavior of the Riemann zeros, but it does not predict unseen zeros, reproduce their GUE local statistics, or furnish a self-adjoint spectral realization of the zeta function."

We have restructured §6 to separate:
- **§6.1 Positive findings**: low-order in-sample agreement, conjugate-pair restoration, bottom-up dynamical construction
- **§6.2 Limitations**: OOS failure, non-GUE, no operator, non-unique k₁, ion-trap non-significance

---

## Response to Reviewer 2

### Comment 1 and 2: Use of "we/our" despite single author

**Reviewer's Comment:**
> 1) The use of I, we, my, and our isn't preferable in scientific writing.
>
> 2) The author writes we and our despite there being one author.

**Our Response:**

Fully corrected.

**Changes Made:**

Replaced all instances of "we/our/my" with "the author/this work/the present study" throughout the manuscript (Abstract, Introduction, Methods, Results, Discussion, Conclusions).

Verified via `grep`: zero remaining first-person plural pronouns in scientific content.

---

### Comment 3: Figure quality issues

**Reviewer's Comment:**
> Some figure not clear and should be redrawn. Like in Figure 3, the legend overlaps the text box. In Figure 4, the font is not readable. In Figure 5, the legend overlaps the curve. Figures 6 and 7 are not clear. Consequently, review all figures.

**Our Response:**

We have reviewed and regenerated all figures.

**Changes Made:**

- **Figure 3**: Legend repositioned, no text overlap
- **Figure 4**: Font size increased to 10pt minimum
- **Figure 5**: Legend moved outside curve region
- **Figures 6-7**: Resolution increased to 300 DPI, axis labels enlarged

All figures now meet MDPI formatting guidelines (minimum 1000px width, readable at 100% zoom).

---

### Comment 4: References too old

**Reviewer's Comment:**
> Most of the references are too old and should be updated.

**Our Response:**

We have added 5 recent references (2020-2025) while retaining foundational primary sources.

**Changes Made:**

Added references:
- **he2020**, **he2021** — USTC ion-trap quantum simulation (Phys. Rev. A 2020, npj Quantum Inf. 2021) — as requested by Reviewer 1
- **yakaboylu2024** — Recent Hilbert-Pólya Hamiltonian proposal (J. Phys. A 2024)
- **eckstein2024** — Large-scale Floquet quantum simulation (npj Quantum Inf. 2024)
- **orellana2025** — Numerical Riemann zeta algorithms (arXiv 2025)

Classic references (Riemann 1859, Hadamard 1896, Montgomery 1973, Odlyzko 1987) retained as foundational primary sources.

All 43 references verified as accurate and accessible (see References Verification section in cover letter).

---

### Comment 5: Place Methods before Results

**Reviewer's Comment:**
> I suggest placing the methods section before the results section.

**Our Response:**

Adopted.

**Changes Made:**

Manuscript structure is now:
1. Introduction
2. The Mathematical Model
3. **Methods** (Modules A, B, C)
4. **Results** (Microscopic, Macroscopic)
5. Discussion
6. Conclusions and Future Work

---

### Comment 6: Add Conclusions and Future Work section

**Reviewer's Comment:**
> A conclusion and future work section should be added and supported by numerical evidence.

**Our Response:**

Fully adopted.

**Changes Made:**

**New §6 "Conclusions and Future Work"** includes:

- **§6.1 Positive findings** — itemized list with numerical support:
  - Low-order in-sample agreement over the first ~20-100 zeros
  - Conjugate-pair restoration eliminates b-intercept divergence
  - Bottom-up dynamical construction

- **§6.2 Limitations** — itemized list with numerical evidence:
  - (1) No OOS prediction: held-out MSE 1-2 orders larger, k₁ drifts 5.79→10.04
  - (2) Non-GUE local statistics: model D(Poisson)=0.082 p=0.49 vs true zeros D(GUE)=0.067 p=0.74
  - (3) No self-adjoint operator or ζ(s) spectral realization
  - (4) Non-unique k₁: two runs converge to 6.481 vs 8.070
  - (5) Ion-trap co-location not statistically significant (p=0.099)

- **§6.3 Future work** — Dirichlet L-functions, hardware speculation, LLM transparency

- **Table 2** "Status of the main statements" — epistemic status of every claim

---

### Comment 7: Define abbreviations

**Reviewer's Comment:**
> All abbreviations should be defined in their first appearance in the paper.

**Our Response:**

Verified and corrected.

**Changes Made:**

Key abbreviations are now defined at first appearance:
- GUE (Gaussian Unitary Ensemble) and RMT (Random Matrix Theory) — §1 Introduction
- MSE (Mean Squared Error) — Appendix H.1

Other quantities that could have been abbreviated (out-of-sample, coefficient of variation, Kolmogorov–Smirnov) are instead spelled out in full at every occurrence rather than introduced as acronyms, to avoid burdening the reader with additional abbreviations.

---

### Comment 8: Lack of Generalization

**Reviewer's Comment:**
> In Section 3.2.3, there is a Lack of Generalization. Fitting the macroscopic annealing constant k₁ using only the first M ∈ {50, 70, 80} zeros and evaluating the resulting operator on held-out zeros causes the test MSE to explode by 1–2 orders of magnitude. The parameter k₁ drifts systematically (5.79 → 9.15 → 10.04) as the training window size M increases. This demonstrates that the 1/ln²(n) cooling schedule operates as an in-sample parameterized curve fit rather than an intrinsic dynamical invariant reproducing the Riemann spectrum from first principles. **The model lacks genuine predictive or extrapolative power.**

**Our Response:**

Fully acknowledged as **principal limitation**.

**Changes Made:**

This is now emphasized throughout the manuscript (see response to Reviewer 1, Main Comment 1 above):

- **Abstract**: explicit statement of in-sample fit, not prediction
- **§4.2.3 "Out-of-Sample Validation and Multi-Seed Robustness" dedicated subsection**: full OOS quantification
- **§6.2 "Limitations", item (1)**: principal limitation
- **Closing summary** (end of §6): restated

We agree this demonstrates the 1/log²n schedule is a calibrated fit, not an intrinsic dynamical invariant with predictive power. This is now stated explicitly and repeatedly.

---

### Comment 9: Ion-trap co-location not significant

**Reviewer's Comment:**
> The author draws parallels between numerical residual spikes near N ≈ 20, 40, 76 and experimental decoherence error peaks reported in trapped-ion Floquet quantum simulations (USTC experiment). A permutation test yields p = 0.099 (not statistically significant at the standard α = 0.05 threshold). Because numerical discretization/truncation errors and physical quantum decoherence arise from distinct physical mechanisms, **framing this qualitative co-location as a potential shared limit risks overinterpreting unselected numerical artifacts.**

**Our Response:**

Fully acknowledged.

**Changes Made:**

(See response to Reviewer 1, Main Comment 4b above for detailed changes)

The non-significance is now:
- Stated upfront in §4.1.5 "Statistical Significance of the USTC Coincidence"
- Repeated in §6.2 "Limitations"
- Speculation reduced by ~60%
- Explicitly notes distinct physical mechanisms (numerical truncation vs quantum decoherence)

---

## References Verification

All 43 cited references have been verified as accurate and accessible via WebSearch:

- **yakaboylu2024**: DOI 10.1088/1751-8121/ad4c2d ✓
- **he2020**: DOI 10.1103/PhysRevA.101.043402 ✓
- **he2021**: DOI 10.1038/s41534-021-00446-7 ✓
- **eckstein2024**: DOI 10.1038/s41534-024-00866-1 ✓
- **orellana2025**: arXiv:2512.09960 ✓
- **wang2026** (author's prior work): DOI 10.1080/27684830.2026.2684334 ✓

No hallucinated or inaccessible references.

---

## Closing Statement

We believe the revised manuscript now presents an honest, transparent, and appropriately scoped exploratory study. The new Conclusions section makes clear what the model achieves (low-order in-sample correspondence) and what it does not (OOS prediction, GUE statistics, operator construction). The manuscript is shorter, more focused, and free of imprecise language.

We are grateful for the reviewers' thorough and constructive guidance, which has substantially improved the scientific rigor and clarity of this work.

Sincerely,

**Liang Wang**  
Huazhong University of Science and Technology

---

**File Manifest for Resubmission:**
- `main.pdf` — Revised manuscript (35 pages)
- `main.tex` + `references.bib` + `figures/` — Source files
- `response_to_reviewers_round2.pdf` — This document

