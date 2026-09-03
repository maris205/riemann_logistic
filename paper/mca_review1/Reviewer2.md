In this work, the authors proposed a Numerical Spectral Correspondence between a Non-Autonomous Quadratic Map and the Riemann Zeros.

The work is good, but I have some comments as follows.

1) The use of I, we, my, and our isn't preferable in scientific writing.

2) The author writes we and our despite there being one author.

3) Some figure not clear and should be redrawn. Like in Figure 3, the legend overlaps the text box. In Figure 4, the font is not readable. In Figure 5, the legend overlaps the curve. Figures 6 and 7 are not clear. Consequently, review all figures.

4) Most of the references are too old and should be updated.

5) I suggest placing the methods section before the results section.

6) A conclusion and future work section should be added and supported by numerical evidence.

7) All abbreviations should be defined in their first appearance in the paper.

8) In Section 3.2.3, there is a Lack of Generalization. Fitting the macroscopic annealing constant k1 using only the first M ∈ {50, 70, 80} zeros and evaluating the resulting operator on held-out zeros causes the test MSE to explode by 1–2 orders of magnitude. The parameter k1 drifts systematically (5.79 → 9.15 → 10.04) as the training window size M increases. This demonstrates that the 1/ln²(n) cooling schedule operates as an in-sample parameterized curve fit rather than an intrinsic dynamical invariant reproducing the Riemann spectrum from first principles. The model lacks genuine predictive or extrapolative power.

9) The author draws parallels between numerical residual spikes near N ≈ 20, 40, 76 and experimental decoherence error peaks reported in trapped-ion Floquet quantum simulations (USTC experiment). A permutation test yields p = 0.099 (not statistically significant at the standard α = 0.05 threshold). Because numerical discretization/truncation errors and physical quantum decoherence arise from distinct physical mechanisms, framing this qualitative co-location as a potential shared limit risks overinterpreting unselected numerical artifacts.