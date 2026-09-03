# Cover Letter — Resubmission of Manuscript mca-4536128

**Manuscript ID:** mca-4536128 (previously mca-4499257)
**Title (revised):** "Numerical Spectral Correspondence between a Non-Autonomous Quadratic Map and the Riemann Zeros: An Exploratory Study"
**Journal:** Mathematical and Computational Applications (MCA)
**Author:** Liang Wang

Dear Editor,

Thank you for the opportunity to revise and resubmit this manuscript. We received four
detailed reviewer reports, three recommending rejection and one major revision. We have
made extensive changes throughout the manuscript in direct response to every itemized
comment, and we attach one point-by-point response letter per reviewer report
(`response_to_reviewer_58231421.md`, `response_to_reviewer_58231636.md`,
`response_to_reviewer_58277902.md`, `response_to_reviewer_58279672.md`).

Rather than argue against the reviewers' central assessment, our revision strategy has
been to accept it: this is a numerical, exploratory study that cannot establish a
Hilbert–Pólya-type operator or a genuinely predictive fit, and several of the original
draft's claims were overstated relative to what the numerics actually show. We have
rewritten the manuscript to state its actual, more modest evidentiary status explicitly
and consistently — including reporting new negative/unfavorable results discovered while
investigating the reviewers' concerns — rather than to defend the original framing.

## Summary of the main changes

**1. Title, abstract, and framing.** The title no longer claims a "spectral isomorphism"
or "renormalization flow"; it now reads as above and the abstract states the results are
"numerical observations on a heuristic model, not a proof or Hilbert–Pólya-type operator
construction." Rhetorical/metaphorical language flagged by reviewers (e.g. "ocean of
chaos," "topological fossil," "spontaneous calibrator," "Linear Homomorphic Mapping,"
"particle-hole symmetry," "global cooling inertia," "Topology Maintenance") has been
removed throughout and replaced with plain technical language.

**2. New epistemic status table (Table "Status of the main claims").** Every main claim in the
paper is now tagged with its actual epistemic status (established/published, numerical
observation, heuristic, qualitative, conjectural, refuted, or not claimed), including two
entries we mark **Refuted**: out-of-sample extrapolation of the fitted k1, and a match to
GUE local (unfolded) spacing statistics.

**3. Out-of-sample validation and multi-seed robustness (new §`sec:oos`).** We added a
genuine held-out test: fitting k1 on the first M∈{50,70,80} zeros and evaluating on the
rest. Held-out MSE is one to two orders of magnitude larger than training MSE at every M,
and we state directly that "the model is a calibrated in-sample reconstruction... not a
predictor of unseen zeros." A ten-seed robustness check at fixed M=70 is also reported.

**4. Statistical significance testing for the N=20 / USTC coincidence (new
§`sec:n20-significance`).** The previously cherry-picked N=20 residual spike is now
reported from 20 unselected eigendecomposition trials, plus a Spearman rank correlation
and a 100,000-resample permutation test for the specific joint-rank coincidence
(p=0.099, not significant at the conventional threshold).

**5. Genuine like-for-like GUE/RMT test (§`sec:gue-comparison`).** We unfolded our
model's own eigenphases and ran a KS test against GUE and Poisson spacing distributions;
unlike the true Riemann zeros (which do match GUE, as expected from the literature), our
model's own eigenphases do **not** match GUE and are statistically indistinguishable from
Poisson. We report this directly as a limitation rather than omitting it.

**6. Disclosure and root-cause tracing of the Table 1 / Figure 4–5 numerical
inconsistency.** We traced the discrepancy between k1≈6.481 (Table 1) and k1≈8.070
(Figures 4–5) to genuine run-to-run instability of an *unseeded* `differential_evolution`
call on a non-convex, multimodal objective — not two different experimental protocols.
Both values are now reported explicitly in the relevant captions, and Table `tab:claims`
carries a corresponding "Refuted" row for run-to-run reproducibility of this optimum.

**7. Four new targeted numerical experiments, added in this final revision round
(new §`sec:post-review-checks`), directly answering four specific reviewer requests for
numerical evidence rather than qualitative acknowledgment.** All four experiments'
code and raw output are included in the submission (`review_experiments/` in the
repository). We report all four results as-is, including where they are unfavorable:
   - *Phase-extraction/Module-C sensitivity*: varying only the eigenvalue magnitude
     cutoff, conjugate-branch selection, and sort-before-unwrap convention on one fixed
     empirical matrix changes the fit MSE by roughly 20–40×, and can flip the fitted
     slope's sign. This confirms, with numbers, a sensitivity we had previously only
     flagged as an unverified assumption.
   - *Gaussian-splatting vs. Monte Carlo overlap*: under matched parameters and a shared
     eigenvalue filter, the two transfer-matrix construction methods retained only 3
     directly-comparable eigenvalues, over which their independently fitted slopes
     differed by roughly 35×. This is a negative/inconclusive result, not the
     consistency check we had hoped to run.
   - *Epsilon out-of-sample split*: selecting the microscopic discretization scale ε on
     half of the first 6 zeros and testing on the other half shows the held-out error is
     roughly 5× larger under an asymmetric split than when ε is chosen with knowledge of
     all 6 zeros — concrete, quantitative target-leakage evidence, on the same footing as
     the existing k1 out-of-sample result.
   - *Multi-seed DE robustness at reduced scale*: rerunning the same optimizer
     configuration under five explicit fixed seeds (at a computationally tractable
     reduced scale) shows k1 ranging from 9.33 to 14.26 (std 1.74), independently
     confirming that the objective is genuinely multimodal rather than a one-off
     discrepancy between the two full-scale runs behind Table 1 and Figures 4–5.

   We want to be transparent with the editor and reviewers that these four experiments
   were designed to test open concerns, and three of the four returned results that
   reinforce rather than resolve the underlying limitations. We report them honestly
   rather than omit or reframe them, consistent with the overall revision strategy above.

**8. Terminology and mathematical corrections.** We fixed a transcription error in the
row-normalization equation for the empirical transition matrix (the implementation was
always correct; only the typeset equation was wrong in the previous draft), corrected
"Feigenbaum critical point" terminology, and removed remaining anthropomorphic/quantum
terminology not warranted by the classical, non-unitary construction actually used.

**9. Reproducibility.** A new Appendix subsection ("Reproducibility Guide: Recommended
Hardware and Experimental Pipeline") and an updated repository `readme.md` give explicit
hardware/runtime requirements and map every numerical claim in the paper to a specific,
runnable script, all of which (including the four new post-review-check scripts) are
released with pre-computed output alongside this submission.

## What remains open

Consistent with our stated revision strategy, several concerns raised by reviewers are
**not** claimed to be resolved: a theoretical justification for treating the
time-averaged empirical matrix's spectrum as representative of the true time-ordered
product's spectrum (raised independently by two reviewers) remains an open, unproven
heuristic; target leakage has not been eliminated for μc or for the separately-refit
N=1000/10000 regimes; and a systematic floating-point/compiler-precision sensitivity
sweep for the full 10¹⁰-step trajectories has not been run. Each response letter states
these limitations explicitly rather than asserting a fix we have not verified.

We believe the manuscript, as revised, now presents an honest and considerably more
carefully hedged account of what this numerical exploration does and does not show, and
we thank the reviewers for pushing the work toward that more accurate framing. We hope
the revised manuscript, together with the attached point-by-point responses, addresses
the Editorial Office's concerns sufficiently for further consideration.

Sincerely,
Liang Wang
