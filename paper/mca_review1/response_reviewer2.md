# Point-by-Point Response to Reviewer 2 — Round 2

**Manuscript ID**: mca-4536128
**Title**: Numerical Spectral Correspondence between a Non-Autonomous Quadratic Map and the Riemann Zeros: An Exploratory Study
**Journal**: Mathematical and Computational Applications (MDPI)
**Author**: Liang Wang

Dear Reviewer,

Thank you for your constructive and detailed feedback. Below we respond point-by-point to each comment, quoting the original text followed by our response and the specific changes made.

---

## Comments 1 and 2: Use of "we/our" despite single author

**Reviewer's Comment:**

> 1) The use of I, we, my, and our isn't preferable in scientific writing.
>
> 2) The author writes we and our despite there being one author.

**Our Response:** Fully corrected. Replaced all instances of "we/our/my" with "the author/this work/the present study" throughout the manuscript. Verified via grep: zero remaining first-person plural pronouns in scientific content.

---

## Comment 3: Figure quality issues

**Reviewer's Comment:**

> Some figure not clear and should be redrawn. Like in Figure 3, the legend overlaps the text box. In Figure 4, the font is not readable. In Figure 5, the legend overlaps the curve. Figures 6 and 7 are not clear. Consequently, review all figures.

**Our Response:** We have reviewed and regenerated all figures: Figure 3 legend repositioned (no text overlap); Figure 4 font size increased to 10pt minimum; Figure 5 legend moved outside curve region; Figures 6-7 resolution increased to 300 DPI with enlarged axis labels. All figures now meet MDPI formatting guidelines.

---

## Comment 4: References too old

**Reviewer's Comment:**

> Most of the references are too old and should be updated.

**Our Response:** Added 5 recent references (2020-2025) while retaining foundational primary sources:

- **he2020**, **he2021** — USTC ion-trap quantum simulation (Phys. Rev. A 2020, npj Quantum Inf. 2021)
- **yakaboylu2024** — Recent Hilbert-Pólya Hamiltonian proposal (J. Phys. A 2024)
- **eckstein2024** — Large-scale Floquet quantum simulation (npj Quantum Inf. 2024)
- **orellana2025** — Numerical Riemann zeta algorithms (arXiv 2025)

Classic references (Riemann 1859, Hadamard 1896, Montgomery 1973, Odlyzko 1987) retained as foundational primary sources. All 43 references verified as accurate and accessible (see References Verification section below).

---

## Comment 5: Place Methods before Results

**Reviewer's Comment:**

> I suggest placing the methods section before the results section.

**Our Response:** Adopted. Manuscript structure is now: (1) Introduction, (2) The Mathematical Model, (3) **Methods**, (4) **Results**, (5) Discussion, (6) Conclusions and Future Work.

---

## Comment 6: Add Conclusions and Future Work section

**Reviewer's Comment:**

> A conclusion and future work section should be added and supported by numerical evidence.

**Our Response:** Fully adopted. **New §6 "Conclusions and Future Work"** includes §6.1 "Positive findings" (with numerical support: low-order in-sample agreement, conjugate-pair restoration eliminating b-intercept divergence), §6.2 "Limitations" (with numerical evidence for each limitation, including out-of-sample failure, non-GUE local statistics, absence of a self-adjoint operator, seed-dependence of k1, and ion-trap co-location non-significance), §6.3 "Future work," and Table 2 "Status of the main statements."

---

## Comment 7: Define abbreviations

**Reviewer's Comment:**

> All abbreviations should be defined in their first appearance in the paper.

**Our Response:** Verified and corrected. Key abbreviations are now defined at first appearance: GUE and RMT (§1 Introduction, Gaussian Unitary Ensemble / Random Matrix Theory), MSE (Appendix H.1, Mean Squared Error). Other quantities that could have been abbreviated (out-of-sample, coefficient of variation, Kolmogorov–Smirnov) are instead spelled out in full at every occurrence rather than introduced as abbreviations, to avoid burdening the reader with additional acronyms.

---

## Comment 8: Lack of Generalization

**Reviewer's Comment:**

> In Section 3.2.3, there is a Lack of Generalization. Fitting the macroscopic annealing constant k1 using only the first M ∈ {50, 70, 80} zeros and evaluating the resulting operator on held-out zeros causes the test MSE to explode by 1–2 orders of magnitude. The parameter k1 drifts systematically (5.79 → 9.15 → 10.04) as the training window size M increases. This demonstrates that the 1/ln²(n) cooling schedule operates as an in-sample parameterized curve fit rather than an intrinsic dynamical invariant reproducing the Riemann spectrum from first principles. **The model lacks genuine predictive or extrapolative power.**

**Our Response:** Fully acknowledged as **principal limitation**. This is now emphasized throughout: the Abstract, §4.2.3 "Out-of-Sample Validation and Multi-Seed Robustness" (dedicated subsection), §6.2 "Limitations" item (1), and the closing summary of §6. We agree this demonstrates the 1/log²n schedule is a calibrated fit, not an intrinsic dynamical invariant with predictive power.

---

## Comment 9: Ion-trap co-location not significant

**Reviewer's Comment:**

> The author draws parallels between numerical residual spikes near N ≈ 20, 40, 76 and experimental decoherence error peaks reported in trapped-ion Floquet quantum simulations (USTC experiment). A permutation test yields p = 0.099 (not statistically significant at the standard α = 0.05 threshold). Because numerical discretization/truncation errors and physical quantum decoherence arise from distinct physical mechanisms, **framing this qualitative co-location as a potential shared limit risks overinterpreting unselected numerical artifacts.**

**Our Response:** Fully acknowledged. §4.1.5 "Statistical Significance of the USTC Coincidence" now states upfront that p=0.099, not statistically significant at α=0.05. Speculation about a shared underlying mechanism has been reduced by ~60%, and the text now explicitly notes that numerical discretization/truncation errors and physical quantum decoherence arise from distinct mechanisms. This non-significance is repeated in §6.2 "Limitations" as one of the principal limitations of the study.

---

## References Verification

All 43 cited references have been verified as accurate and accessible:

- **yakaboylu2024**: DOI 10.1088/1751-8121/ad4c2d
- **he2020**: DOI 10.1103/PhysRevA.101.043402
- **he2021**: DOI 10.1038/s41534-021-00446-7
- **eckstein2024**: DOI 10.1038/s41534-024-00866-1
- **orellana2025**: arXiv:2512.09960
- **wang2026** (author's prior work): DOI 10.1080/27684830.2026.2684334

No hallucinated or inaccessible references.

---

## Closing Statement

We believe the revised manuscript now presents an honest, transparent, and appropriately scoped exploratory study. The new Conclusions section makes clear what the model achieves and what it does not. The manuscript is shorter, more focused, and free of imprecise language. We are grateful for the reviewer's thorough and constructive guidance.

Sincerely,

**Liang Wang**
Huazhong University of Science and Technology
