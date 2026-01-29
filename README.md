---

### Conditional Inverse Learning of Time-Varying Reproduction Numbers Inference (CIRL)

This repository provides the reference implementation for **Conditional Inverse Reproduction Learning (CIRL)**, a physics-informed statistical learning framework for estimating time-varying reproduction numbers ( R_t ) from epidemic incidence data under renewal dynamics.

The code supports both **simulated epidemics** and **real-world surveillance data**, and includes implementations for reproduction number inference as well as forward reconstruction of incidence trajectories.

---

### Data Sources

All datasets used in this repository are derived from **publicly available sources** that have been widely used in the epidemiological modeling literature.

#### 1. Simulated Epidemic Data

Synthetic epidemic data are generated following the simulation framework described in:

> Parag, K. V., & Donnelly, C. A. (2022).
> *Using information theory to optimise epidemic models for real-time prediction and control*.
> **PLoS Computational Biology**, 18(11), e1011021.
> DOI: [https://doi.org/10.1371/journal.pcbi.1011021](https://doi.org/10.1371/journal.pcbi.1011021)

These simulations are used to evaluate estimation accuracy, responsiveness to regime changes, and robustness under controlled conditions.

---

#### 2. SARS (2003, Hong Kong)

The real-world SARS incidence data for Hong Kong (2003) are obtained from:

> Cori, A., Ferguson, N. M., Fraser, C., & Cauchemez, S. (2013).
> *A new framework and software to estimate time-varying reproduction numbers during epidemics*.
> **American Journal of Epidemiology**, 178(9), 1505–1512.
> DOI: [https://doi.org/10.1093/aje/kwt133](https://doi.org/10.1093/aje/kwt133)

This dataset is commonly used as a benchmark for validating time-varying ( R_t ) estimation methods.

---

#### 3. COVID-19 (2020, Italy – Ontario Level)

The COVID-19 incidence data for Italy (Ontario-level aggregation, 2020) are sourced from:

> Wang, L., Zhang, Y., & Wang, X. (2022).
> *Estimating time-varying reproduction number by deep learning techniques*.
> **Chaos, Solitons & Fractals**, 157, 111947.
> DOI: [https://doi.org/10.1016/j.chaos.2022.111947](https://doi.org/10.1016/j.chaos.2022.111947)

This dataset is used to demonstrate CIRL on real epidemic data characterized by reporting noise, over-dispersion, and non-stationary transmission dynamics.

---

### Baseline Methods

Baseline and comparison methods implemented in this repository follow the experimental setup described in:

> Wang, L., Zhang, Y., & Wang, X. (2022).
> *Estimating time-varying reproduction number by deep learning techniques*.
> **Chaos, Solitons & Fractals**, 157, 111947.
> DOI: [https://doi.org/10.1016/j.chaos.2022.111947](https://doi.org/10.1016/j.chaos.2022.111947)

Where applicable, baseline implementations are adapted from publicly released code or re-implemented to ensure consistency of experimental conditions.

---

### Code Structure

* `data/`
  Processed incidence data and simulation scripts.
* `models/`
  Conditional inverse mapping network and observation models.
* `train/`
  Training and evaluation pipelines.
* `experiments/`
  Scripts for reproduction number inference and incidence reconstruction.
* `baselines/`
  Implementations of comparison methods.

---

### Usage Example

This repository provides an end-to-end example for estimating ( R_t ) and reconstructing ( I_t ) using COVID-19 (Italy, 2020) data:

```bash
python run_cirl_covid_italy.py
```

The script outputs:

* Estimated time-varying reproduction numbers ( \hat{R}_t )
* Reconstructed incidence trajectories ( \hat{I}_t )

---

### Disclaimer

This repository is provided for **research and reproducibility purposes only**.
All data sources are publicly available and credited to their original authors.

---
