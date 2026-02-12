# Riemann-Dynamics-AI: Spectral Isomorphism Discovery

This repository contains the numerical experiments and dynamical systems identified during our **AI-augmented search** for the physical origins of the **Riemann Zeros**. We demonstrate that the distribution of the first 10,000 zeros can be effectively reconstructed by a specific **non-autonomous renormalization flow**.

# Riemann-Dynamics-AI: Spectral Isomorphism Discovery

> **"Wait... It's HIM!!!"**
> 
> While using AI to scale the search for dynamical systems isomorphic to the **Riemann Zeros**, I was deeply immersed in the **Berry-Keating Conjecture**. Suddenly, it struck me: the Sir Michael Berry who had been kindly replying to my emails is the same legend who "legislated" the spectral statistics of the zeros (and famously made a frog fly)! **Oh my!!!** > 
> This project is a direct tribute to that realization.

![Spectral Alignment Results](./fig2.png)
*Figure 2: Deterministic trajectory alignment with R² > 0.997 achieved via AI-augmented search.*


## 📊 Quick Start: Visualizing Results

If you want to reproduce the result plots (like `fig2.png`) immediately:

1. **Clone the full repository** to your local machine.
2. **Locate the compressed data** in the `riemann_10k_survey` directory and **unzip it** first.
3. **Run `phase_unwrapping.ipynb**`: This is the primary pre-processing script for handling the spectral phase of the zeta function and generating the core visual alignments.

---

## 📁 Repository Structure (Ordered by Workflow)

1. **`phase_unwrapping.ipynb`**: Pre-processing scripts for handling the spectral phase of the zeta function. **Start here for visualization.**
2. **`riemann_10k_survey.ipynb`**: Comparative analysis against GUE (Gaussian Unitary Ensemble) statistics and final performance metrics.
3. **`riemann_10k_harvest.ipynb`**: The "harvesting" pipeline where the AI-guided symbolic regression surfaced the optimal parameters.
4. **`pipeline_survey.py`**: The automated search core used to scale the dynamical isomorphism discovery.
5. **`riemann_10k_true.npy`**: Ground truth data for the first 10,000 Riemann zeros used for calibration.

---

## ⚡ Heavy Computation: Rerunning the Search

If you wish to re-run the entire discovery pipeline, please note that it is computationally intensive. **A CPU with high core count (e.g., 256 cores) is highly recommended.**

### Step-by-Step Execution:

1. **Environment Setup**: Ensure you have Python 3.8+ and the required deep learning/dynamics libraries installed.
2. **Data Preparation**: Ensure `riemann_10k_true.npy` is in the root directory.
3. **Scalable Search**: Run `python pipeline_survey.py` to initiate the parallelized search for the optimal renormalization flow parameters.
4. **Parameter Harvesting**: Use `riemann_10k_harvest.ipynb` to filter and extract the most promising candidate formulas from the search logs.

---

## 🖼️ Result Preview
![拟合图](fig2.png)

Our model achieves **Deterministic Spectral Matching** with :

> "Success = My Idea. Failure = Gemini 3's Hallucination." 😂

## cite paper
wang, . liang . (2026). Spectral Isomorphism between Renormalization Flow in Non-Autonomous Quadratic Maps and Riemann Zeros (v1.0). Zenodo. https://doi.org/10.5281/zenodo.18618087




