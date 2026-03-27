# NAF-BMA
Supplementary code for **"Neural Autoregressive Flows based Variational Bayes Model Averaging"**.

This repository contains the code used to reproduce the simulation studies and real-data experiments in the paper. The implementation is primarily in Python, with a small number of R scripts for selected examples.

## Repository structure

### Top-level files

| File | Purpose |
|---|---|
| `README.md` | Repository overview and usage notes. |
| `VI_models.py` | Core variational inference and Bayesian model averaging routines. |
| `flow_models.py` | Main flow-based model definitions used in the NAF-BMA implementation. |
| `simulated_data_generation.py` | Data generation utilities for the simulation studies. |
| `lm_simulation.py` | Linear regression simulation study for the main manuscript. |
| `lm_mc_check_simulation.py` | Additional linear regression simulation checks, including Monte Carlo-related diagnostics. |
| `logistic_naf_simulation.py` | Logistic regression simulation study using NAF-BMA. |
| `logistic_mcmc_vi_simulation.py` | Logistic regression comparison script for MCMC / variational baselines. |
| `gmm_bma_simulation.py` | Gaussian mixture model experiment for the finite mixture model example. |
| `experiments_USCrime.py` | U.S. Crime real-data analysis. |
| `experiments_nuclear.py` | Nuclear mass prediction real-data analysis. |
| `utils.py` | Shared helper functions used across experiments and model code. |

### `R_code/`

| File | Purpose |
|---|---|
| `R_code.Rproj` | RStudio project file for the R scripts in this folder. |
| `Toy_data_LM.R` | R script for the toy linear regression example. |
| `uscrime_R.R` | R script for the U.S. Crime example in R. |

### `torchkit/`

`torchkit` is a supporting PyTorch toolkit that contains reusable flow layers, helper functions, and toy experiment modules. This is a modification of the NAF Repository (https://github.com/CW-Huang/torchkit) for experiments for the Neural Autoregressive Flows paper by Huang et al. (https://arxiv.org/abs/1804.00779) The code was adjusted for Python 3.

## Reproducing the paper

The main experiment scripts are:

- `lm_simulation.py` and `logistic_naf_simulation.py` for the simulation studies,
- `gmm_bma_simulation.py` for the finite mixture model example,
- `experiments_USCrime.py` for the U.S. Crime data analysis,
- `experiments_nuclear.py` for the nuclear mass prediction example.

The R scripts in `R_code/` provide additional implementations for selected examples.

## Notes

- Some scripts are written for specific model classes and may require small changes when adapting them to new problems.
- The `torchkit` package contains reusable building blocks and can be extended for additional flow-based models.


## Citation

If you use this code, please cite the associated paper:

**Neural Autoregressive Flows based Variational Bayes Model Averaging**
