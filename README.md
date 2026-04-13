
### An Ising Model Approach to Financial Markets -  Market Crashes as Phase Transitions (Warwick Mathematical Finance Magazine Submission)


## Overview

This project investigates financial market dynamics through the framework of **statistical mechanics**, using the Ising model as a minimal agent-based system.

By modelling traders as interacting spins on a lattice, the project explores how **local interactions** can give rise to **emergent macroscopic behaviour**, including:

* **Large-scale coordination** (market trends)
* **Abrupt regime shifts** (crash-like behaviour)
* **Heavy-tailed return distributions**

The central aim is to assess whether key **stylised facts of financial markets** can be reproduced near criticality.

---

## Model Description

We consider a 2D Ising model on a square lattice with periodic boundary conditions:

* **Spins:** $s_i \in \{-1, +1\}$
* **Hamiltonian:**

$$H = -J \sum_{\langle i,j \rangle} s_i s_j - h \sum_i s_i$$

* **Dynamics:**
    * Metropolis-Hastings updates (standard Ising)
    * Heat-bath dynamics with Bornholdt-type feedback

### Financial Interpretation

| Physics Quantity | Financial Analogy |
| :--- | :--- |
| Spin $s_i$ | Trader decision (buy/sell) |
| Magnetisation $m_t$ | Aggregate market sentiment |
| Temperature $T$ | Market uncertainty |
| Coupling $J$ | Interaction strength between agents |
| Phase transition | Market instability / crashes |

---

## Bornholdt Extension

To introduce feedback effects, we include a global term in the effective field:

$$h_i^{\text{eff}} = J \sum_{\text{nn}} s_j - \alpha s_i |m|$$

This term penalises alignment with the global state when the market is over-extended, generating:
* Intermittent switching
* Increased volatility
* More realistic return dynamics

---

## Synthetic Returns

We define a toy return series derived from the change in magnetisation:

$$r_t = m_{t+1} - m_t$$

This enables analysis of return distributions, volatility dynamics, and autocorrelation structures.

---

## Key Results

* **Near the critical temperature ($T_c$):**
    * Large fluctuations in magnetisation.
    * Fat-tailed return distributions (non-Gaussian).
* **Bornholdt dynamics:**
    * Increased intermittency.
    * Richer volatility structure.
* **Limitations:**
    * Limited long-memory effects.
    * Weak volatility clustering compared to real-world high-frequency data.

---

## Repository Structure

```text
ising-market-model/
│
├── notebooks/          # Exploratory analysis and experiments
│
├── src/
│   ├── ising.py        # Core Ising model and dynamics
│   ├── analysis.py     # Statistical analysis of simulation output
│   ├── master.py       # Experiment orchestration (parameter sweeps)
│   └── plotting.py     # Visualisation tools
│
├── figures/            # Generated plots for analysis and paper
├── paper/              # Research article (PDF + LaTeX source)
│
├── requirements.txt
└── README.md

---
```

### Installation

**Clone the repository:**

```bash
git clone https://github.com/yourusername/ising-market-model.git
cd ising-market-model
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

---

### Usage

## Running an Experiment

```python
from src.master import IsingExperiment

exp = IsingExperiment(
    size=50,
    temperatures=[1.5, 2.0, 2.269, 2.5],
    alphas=[1.0, 2.0, 4.0],
    update_rule="bornholdt",
    seed=42,
)

results = exp.run_grid(
    n_equil=5000,
    n_steps=10000,
    sample_freq=10
)
```

### Visualisation

```python
from src.plotting import IsingPlotter

plotter = IsingPlotter(results)

plotter.plot_magnetisation_vs_T()
plotter.plot_susceptibility()

# Plot specific dynamics
plotter.plot_time_series(T=2.269, alpha=2.0)
plotter.plot_returns(T=2.269, alpha=2.0)
plotter.plot_acf(T=2.269, alpha=2.0)
```

---

## Analysis Pipeline

| Step | Module | Description |
|------|--------|-------------|
| 1 | `ising.py` | Generates magnetisation and energy time series |
| 2 | `analysis.py` | Computes returns, volatility, autocorrelation, and thermodynamic observables (Binder cumulant, heat capacity) |
| 3 | `master.py` | Runs parameter sweeps across temperature $T$ and feedback strength $\alpha$ |
| 4 | `plotting.py` | Produces publication-quality figures |

---

## Future Work

- [ ] Research Paper to be uploaded 
- [ ] Time-varying temperature (market regimes)
- [ ] Heterogeneous agent models
- [ ] Calibration to real market data (e.g. S&P 500)
- [ ] Information-theoretic measures (entropy, KL divergence)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- Statistical mechanics and Ising model literature
- Econophysics research on financial markets
- Python scientific computing ecosystem
