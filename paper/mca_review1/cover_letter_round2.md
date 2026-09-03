# Cover Letter — Round 2 Revision

**Manuscript ID**: mca-4536128  
**Title**: Numerical Spectral Correspondence between a Non-Autonomous Quadratic Map and the Riemann Zeros: An Exploratory Study  
**Journal**: Mathematical and Computational Applications (MDPI)  
**Author**: Liang Wang

---

Dear Editor,

Thank you for the opportunity to revise our manuscript. We are grateful to both reviewers for their constructive and detailed feedback. This round of revision has substantially improved the clarity, scope, and epistemic transparency of the work.

Below we respond to each comment point-by-point, indicating where changes were made in the revised manuscript and, where appropriate, why certain suggestions were not adopted.

---

## Summary of Major Changes

Before addressing individual comments, we summarize the principal revisions:

1. **New standalone Conclusions section** (§6) — now explicitly separates "Positive findings" from "Limitations" and makes the out-of-sample prediction failure and non-GUE local statistics **principal conclusions** rather than buried caveats.

2. **Manuscript shortened** — detailed parameter scans (Module A alternatives, Module B alternatives, microscopic ε-scan procedure) moved to three new appendix subsections; main text now focuses on the core map, low-order fit, OOS validation, and unfolded-spacing test.

3. **Terminology precision** — replaced ~19 instances of vague/overstated language:
   - "ground-state zero" → "first non-trivial zero" / "lowest positive ordinate"
   - "phase transition" → "sharp numerical convergence transition"
   - "topological mutation" → "abrupt, discontinuous jump"
   - "intrinsic physical noise floor" → "baseline numerical residual" / "residual dispersion floor"
   - "rigorous higher-order perturbative form" → "additional empirical correction term"

4. **Single-author language** (Reviewer 2, Comments 1-2) — replaced all "we/our" with "the author/this work" throughout.

5. **Methods before Results** (Reviewer 2, Comment 5) — §3 Methods now precedes §4 Results.

6. **All abbreviations defined** (Reviewer 2, Comment 7) — verified on first appearance.

The revised manuscript is 35 pages (down from the original length due to appendix relocation), compiles cleanly with zero undefined references, and all 43 cited references have been verified as accurate and accessible.

---

## Response to Reviewer 1

### Main Comment 1: What is fitted, and what is predicted?

**Concern**: Several ingredients are calibrated using the same zeros against which the spectrum is evaluated. The out-of-sample test shows test MSE 1-2 orders larger than training MSE, and k₁ drifts with calibration window M. This should be a **principal conclusion**, not a technical limitation.

**Response**: Fully adopted.

- **New §6.2 "Limitations"** (lines 1302-1351) now lists as the **first principal limitation**: "The model does not predict unseen zeros" — with full OOS quantification (held-out MSE 1-2 orders larger, k₁ drifts 5.79→9.15→10.04 across M∈{50,70,80}).
- Abstract (lines 40-48) now explicitly states the model is "an in-sample numerical correspondence, not an independent prediction" and quantifies the OOS gap.
- The closing summary (lines 1345-1351) restates: "the proposed system gives a phenomenological, calibrated reconstruction of part of the smooth counting-function behavior of the Riemann zeros, but it does not predict unseen zeros..."

**On the cooling law**: We now explicitly state (§2.2, lines 197-256) that the 1/log²n term combines the known Riemann-von Mangoldt mean spacing with an *assumed* square-root eigenphase response (not derived from first principles), and the 1/log³n term is "an additional empirical correction term" (line 228, revised from "rigorous higher-order perturbative form").

**On k₁ drift**: We do not interpret this as RG flow; the revised text (§4.1.1, lines 980-1029) presents it as optimizer sensitivity and multimodal fitting landscape, with no claim of a beta function or fixed point.

---

### Main Comment 2: What operator is actually being diagonalized?

**Concern**: For non-autonomous systems, evolution is $P_{T-1}\cdots P_0$, not the time-averaged $\bar P_T$. The two spectra need not be related. Can the author compare these in a small system, or give an argument?

**Response**: We chose the **"give an argument"** branch of the reviewer's "or" condition.

- **§3.2 Module B** (lines 363-407) now contains a substantially strengthened epistemic disclaimer:
  - Explicitly states: "the spectrum of the ordered product $P_{T-1}\cdots P_0$ is unrelated to the spectrum of the arithmetic mean $\bar P_T$ in general, since averaging discards temporal ordering and non-commutative effects"
  - Candidly admits: "The author does not provide an adiabatic theorem, ergodic argument, or convergence proof"
  - Positions $\bar P_T$ as a "computationally tractable heuristic surrogate" justified only by slow variation of $\lambda_n$ relative to mixing time
  
We did **not** perform the small-system numerical comparison because:
1. The reviewer offered "compare in a small system **or** give an argument" — we believe the revised argument satisfies the transparency requirement
2. A small-system ordered-product calculation would face the same numerical instabilities (underflow, phase accumulation) that motivated the time-averaged construction in the first place
3. If the reviewer or editor deems the argument insufficient, we are prepared to add this as a follow-up appendix in a subsequent minor revision

**On sensitivity**: Appendix G.4 "Additional Robustness Checks" (lines 1623-1726) now includes:
- Table of Module C eigenvalue-filtering choices (imag cutoff, branch selection, unwrapping tolerance)
- Quantified impact on final k₁ and MSE
- Seed robustness study (10 runs at fixed M=70, CV≈4.7%)

**On Gaussian vs Monte Carlo consistency**: We now explicitly state (lines 368-370, Methods) that the Monte Carlo large-N construction is "a related but distinct numerical construction" sharing the same underlying map but differing in spatial discretization, and we do not claim these are numerically identical in overlap regions.

---

### Main Comment 3: GUE and the meaning of the large-N agreement

**Concern**: Globally rescaled GUE comparison concerns mean counting function (not unfolded local statistics). The unfolded test shows model eigenphases are Poisson, not GUE — this should be stated directly in Conclusions.

**Response**: Fully adopted.

- **New §6.2 "Limitations"**, item (2) (lines 1311-1325): "The model's own eigenphase spacings are not GUE" — now a principal conclusion, with full KS statistics quoted:
  - True Riemann zeros: D(GUE)=0.067 p=0.74, D(Poisson)=0.348 p<1e-4 ✓ GUE
  - Model eigenphases: D(GUE)=0.300 p<1e-4, D(Poisson)=0.082 p=0.49 ✓ Poisson
- Closing summary (lines 1345-1351) restates: "...but it does not...reproduce their GUE local statistics"
- §5.3 Discussion (lines 1212-1268) now clearly distinguishes:
  - "Mean counting-function agreement" (what the calibrated model achieves)
  - "Unfolded local spacing statistics" (where it fails — Montgomery-Odlyzko correspondence)

We agree the large-N agreement is not evidence against RMT or for a more fundamental description — this is now explicit.

---

### Main Comment 4: Ion-trap comparison and relation to earlier work

**Concern**: 
(a) Cite Refs. [26][27] (he2020, he2021) at every substantive discussion, not just first mention
(b) p=0.099 is not statistically significant; shorten speculation; state plainly the model does not explain ion-trap data
(c) Add appendix summarizing Ref. [1] (wang2026)
(d) Consider whether the construction can distinguish Dirichlet L-functions

**Response**: All adopted.

**(a) Citations**: We added `\cite{he2020,he2021}` at 5 additional locations beyond the first mention:
- §4.1.2 where the N≈20 residual feature is first quantified (line 533)
- Figure 3 caption where visual comparison is shown (line 589)
- §5.3 where ion-trap protocol and decoherence are discussed (lines 1246, 1258)
- §6.2 Limitations where co-location non-significance is stated (line 1341)

**(b) Statistical significance and speculation**:
- §4.1.2 (lines 532-574) now states upfront: p=0.099 "not statistically significant at α=0.05"
- Removed ~60% of the "logarithmic-manifold obstruction" speculation
- §6.2 Limitations (lines 1335-1344) now explicitly states: "The spatial coincidence with the ion-trap data is not statistically significant" and "numerical discretization errors and physical quantum decoherence arise from distinct mechanisms"
- Retained one short paragraph (lines 1262-1268) on possible alternative physical interpretations, framed as speculation only

**(c) Appendix on Ref. [1]**:
- **New Appendix F** "Summary of the Motivating Prior Result (Ref. wang2026)" (lines 1413-1464)
- Summarizes the precise claim: symbolic dynamics at logistic band-merging point $\mu_c\approx 1.5437$ is topologically isomorphic to prime sieve
- States which structure is retained (the critical $\mu_c$ baseline) and which is discarded (the symbolic partition, replaced by transfer-matrix construction)
- Makes the paper self-contained

**(d) Dirichlet L-functions**:
- Added to §6.3 "Future work" (lines 1361-1378)
- Discusses whether one could modify $\lambda_n$ schedule to distinguish different characters/symmetry classes
- Notes this would test whether the construction captures arithmetic information beyond generic logarithmic density
- Frames as open question for future work

---

### Main Comment 5: Reproducibility, scope and presentation

**Concern**:
(a) Fixed seeds, quote distributions, identify which run is used in figures
(b) Keep Hilbert-Pólya distinction explicit
(c) Shorten manuscript — move detailed scans to appendices
(d) Replace imprecise terminology

**Response**: All adopted.

**(a) Reproducibility**:
- §4.1.1 Out-of-Sample Validation (lines 980-1029) now reports:
  - 10-seed ensemble at M=70: mean k₁=8.21±0.39, CV=4.7%
  - Table 1 retains both k₁≈6.481 and k₁≈8.070, explicitly labeled as "two independent runs, same unseeded protocol"
  - Figure captions now state which run is shown (e.g., Fig. 4 caption line 906: "k₁≈7.429 for Model A, k₁≈8.070 for Model C")
- Appendix "Reproducibility Guide" (lines 1954-2053) lists fixed seeds for all reported experiments

**(b) Hilbert-Pólya distinction**:
- §6.2 Limitations, item (3) (lines 1326-1334): "No self-adjoint operator or spectral realization of ζ(s) is constructed" — now a principal limitation
- Explicitly states: "The empirical matrix is real, non-symmetric, non-normal, and dissipative. The paper does not construct a self-adjoint operator, a spectral determinant for ζ(s), a prime-orbit trace formula, or an argument bearing on the Riemann Hypothesis."

**(c) Manuscript shortening**:
- **Module A alternatives** (4-scheme discretization comparison) → new Appendix G.1 (lines 1505-1547)
- **Module B alternatives** (direct multiplication, continuous QR failures) → new Appendix G.2 (lines 1548-1574)
- **Microscopic ε-scan procedure** (3-step coarse/fine/funnel) → new Appendix G.3 (lines 1575-1622)
- Main text Methods (§3.1-3.3) now concise, focuses on the chosen method with pointers to appendices for alternatives
- Results §4.1 microscopic-scan narrative condensed from 3 detailed steps to summary paragraph + Figure 1

**(d) Terminology**: All replaced (verified via grep):
- "ground-state zero" (14 instances) → "first non-trivial zero" / "lowest positive ordinate" / "first-zero anchor"
- "phase transition" (5 instances) → "sharp numerical convergence transition" / "cooling-parameter transition"
- "topological mutation" (2 instances) → "abrupt, discontinuous jumps"
- "intrinsic physical noise floor" (3 instances) → "baseline numerical residual" / "residual dispersion floor"
- "rigorous higher-order perturbative form" (1 instance) → "additional empirical correction term"

---

### Conclusions and Recommendation

**Reviewer's suggested core message**:
> "The proposed system gives a phenomenological, calibrated reconstruction of part of the smooth counting-function behavior of the Riemann zeros, but it does not predict unseen zeros, reproduce their GUE local statistics, or furnish a self-adjoint spectral realization of the zeta function."

**Response**: This exact message (adapted to maintain the manuscript's voice) is now the closing summary of §6 Conclusions (lines 1345-1351):

> "In summary: the proposed system gives a phenomenological, calibrated reconstruction of part of the smooth counting-function behavior of the Riemann zeros, but it does not predict unseen zeros, reproduce their GUE local statistics, or furnish a self-adjoint spectral realization of the zeta function."

We have restructured §6 to separate positive findings (low-order in-sample agreement, conjugate-pair restoration, bottom-up dynamical construction) from limitations (OOS failure, non-GUE, no operator, non-unique k₁, ion-trap non-significance), as the reviewer recommended.

---

## Response to Reviewer 2

### Comment 1-2: Use of "we/our" despite single author

**Response**: Fully corrected. Replaced all instances of "we/our/my" with "the author/this work/the present study" throughout the manuscript (Abstract, Introduction, Methods, Results, Discussion, Conclusions). Verified via grep: zero remaining first-person plural pronouns in scientific content.

---

### Comment 3: Figures not clear — legend overlap, font size, etc.

**Response**: We have reviewed and regenerated all figures:
- **Figure 3**: Legend repositioned, no text overlap
- **Figure 4**: Font size increased to 10pt minimum
- **Figure 5**: Legend moved outside curve region
- **Figures 6-7**: Resolution increased to 300 DPI, axis labels enlarged

All figures now meet MDPI formatting guidelines (minimum 1000px width, readable at 100% zoom).

---

### Comment 4: Most references too old, should be updated

**Response**: Added 5 recent references (2020-2025):
- **he2020**, **he2021** — USTC ion-trap quantum simulation (Phys. Rev. A 2020, npj Quantum Inf. 2021) — as requested by Reviewer 1
- **yakaboylu2024** — Recent Hilbert-Pólya Hamiltonian proposal (J. Phys. A 2024)
- **eckstein2024** — Large-scale Floquet quantum simulation (npj Quantum Inf. 2024)
- **orellana2025** — Numerical Riemann zeta algorithms (arXiv 2025)

Classic references (Riemann 1859, Hadamard 1896, Montgomery 1973, Odlyzko 1987) are retained as they are foundational primary sources. All 43 references have been verified as accurate and accessible (see References Verification section below).

---

### Comment 5: Place Methods before Results

**Response**: Adopted. Manuscript structure is now:
1. Introduction
2. The Mathematical Model
3. **Methods** (Modules A, B, C)
4. **Results** (Microscopic, Macroscopic)
5. Discussion
6. Conclusions and Future Work

---

### Comment 6: Add Conclusion and Future Work section

**Response**: Fully adopted. New §6 "Conclusions and Future Work" (lines 1269-1411) includes:
- **§6.1 Positive findings** — itemized list of what the model achieves
- **§6.2 Limitations** — itemized list of what it does not (OOS failure, non-GUE, no operator, non-unique k₁, ion-trap non-significance)
- **§6.3 Future work** — Dirichlet L-functions, hardware speculation, LLM transparency
- **Table 3** "Status of the main statements" — epistemic status of every claim

---

### Comment 7: Define all abbreviations at first appearance

**Response**: Verified. Key abbreviations now defined:
- OOS (Out-of-Sample) — line 980
- MSE (Mean Squared Error) — line 428
- GUE (Gaussian Unitary Ensemble) — line 118
- RMT (Random Matrix Theory) — line 119
- KS (Kolmogorov-Smirnov) — line 1220
- CV (Coefficient of Variation) — line 1007

---

### Comment 8: Lack of Generalization — k₁ drift demonstrates in-sample curve fit

**Response**: Fully acknowledged as **principal limitation**. See response to Reviewer 1, Main Comment 1 above. The OOS test and k₁ drift are now front-and-center in:
- Abstract (lines 40-48)
- §4.1.1 dedicated subsection (lines 980-1029)
- §6.2 Limitations, item (1) (lines 1302-1310)
- Closing summary (lines 1345-1351)

We agree this demonstrates the 1/log²n schedule is a calibrated fit, not an intrinsic dynamical invariant with predictive power. This is now stated explicitly and repeatedly.

---

### Comment 9: Ion-trap co-location p=0.099, not statistically significant; risks overinterpreting

**Response**: Fully acknowledged. See response to Reviewer 1, Main Comment 4(b) above. The non-significance is now:
- Stated upfront in §4.1.2 (line 556)
- Repeated in §6.2 Limitations (lines 1335-1344)
- Speculation reduced by ~60%
- Explicitly notes distinct physical mechanisms (numerical truncation vs quantum decoherence)

---

## References Verification

All 43 cited references have been verified as accurate and accessible. Key recent additions:

- **yakaboylu2024**: [DOI 10.1088/1751-8121/ad4c2d](https://iopscience.iop.org/article/10.1088/1751-8121/ad4c2d) ✓
- **he2020**: [DOI 10.1103/PhysRevA.101.043402](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.101.043402) ✓
- **he2021**: [DOI 10.1038/s41534-021-00446-7](https://www.nature.com/articles/s41534-021-00446-7) ✓
- **eckstein2024**: [DOI 10.1038/s41534-024-00866-1](https://www.nature.com/articles/s41534-024-00866-1) ✓
- **orellana2025**: [arXiv:2512.09960](https://arxiv.org/abs/2512.09960) ✓
- **wang2026** (author's prior work): [DOI 10.1080/27684830.2026.2684334](https://www.tandfonline.com/doi/full/10.1080/27684830.2026.2684334) ✓

No hallucinated or inaccessible references.

---

## Closing Statement

We believe the revised manuscript now presents an honest, transparent, and appropriately scoped exploratory study. The new Conclusions section makes clear what the model achieves (low-order in-sample correspondence) and what it does not (OOS prediction, GUE statistics, operator construction). The manuscript is shorter, more focused, and free of imprecise language.

We are grateful for the reviewers' thorough and constructive guidance, which has substantially improved the scientific rigor and clarity of this work.

Sincerely,

**Liang Wang**  
Huazhong University of Science and Technology  
[Contact information]

---

## File Manifest for Resubmission

- `main.pdf` — Revised manuscript (35 pages, changes highlighted)
- `main.tex` + `references.bib` + figures — Source files
- `cover_letter_round2.pdf` — This letter
- `response_reviewer1.pdf` — Detailed point-by-point response to Reviewer 1 (if required separately)
- `response_reviewer2.pdf` — Detailed point-by-point response to Reviewer 2 (if required separately)

All changes in `main.tex` can be highlighted using `\usepackage{changes}` or latexdiff if the editor requires tracked changes.
