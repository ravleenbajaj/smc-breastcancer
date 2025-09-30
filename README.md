# Annealed Sequential Monte Carlo for Bayesian Logistic Regression

A Julia implementation of Annealed Sequential Monte Carlo (SMC) for Bayesian inference on logistic regression, applied to the Wisconsin Breast Cancer dataset.

## Overview

This project demonstrates how to use Sequential Monte Carlo with annealing to sample from posterior distributions of logistic regression coefficients. The algorithm gradually transitions from the prior distribution (β=0) to the posterior distribution (β=1), making it particularly effective for complex, multimodal posteriors.

## Features

- **Annealed SMC Implementation**: Adaptive temperature scheduling to maintain effective sample size
- **Bayesian Logistic Regression**: Full posterior inference for classification problems
- **Comprehensive Diagnostics**: Acceptance rates, ESS monitoring, and particle weight distributions
- **Visualization Suite**: Posterior distributions, annealing schedules, and joint posterior plots
- **Medical Application**: Classification of breast cancer tumors (malignant vs benign)

## Dataset

The Wisconsin Breast Cancer dataset contains 699 samples with 9 features measuring cellular characteristics:

- Clump Thickness
- Uniformity of Cell Size
- Uniformity of Cell Shape
- Marginal Adhesion
- Single Epithelial Cell Size
- Bare Nuclei
- Bland Chromatin
- Normal Nucleoli
- Mitoses

**Target**: Binary classification (Benign: 458 samples, Malignant: 241 samples)

## Requirements

```julia
using Pkg
Pkg.add(["CSV", "DataFrames", "Printf", "Plots", "StatsBase", "HTTP", "Distributions"])
```

## Project Structure

```
smc-breastcancer/
├── analysis.jl          # Main analysis script
├── smc.jl              # SMC algorithm implementation
├── dataset.jl          # Data loading utilities
└── README.md
```

## Usage

```julia
# Load required modules
include("dataset.jl")
include("smc.jl")
using .SMC

# Load data
X, y = load_breast_cancer()

# Configure SMC parameters
N_particles = 500
mcmc_steps = 5
step_scale = 0.05
ess_threshold = 0.5

# Run annealed SMC
particles, particle_weights, betas, acc_hist = SMC.annealed_smc(
    X, y;
    N=N_particles,
    mcmc_steps=mcmc_steps,
    step_scale=step_scale,
    ess_frac=ess_threshold
)
```

## Key Results

### Model Performance
- **Accuracy**: 86.41%
- **Sensitivity**: 0.842
- **Specificity**: 0.876
- **Precision**: 0.781

### Top Predictive Features

Features with 95% credible intervals excluding zero:

1. **Cell Size** (β = 0.912 ± 0.129) - Increases malignancy risk
2. **Epithelial Cell Size** (β = -0.818 ± 0.106) - Decreases malignancy risk
3. **Bare Nuclei** (β = 0.563 ± 0.058) - Increases malignancy risk
4. **Bland Chromatin** (β = -0.521 ± 0.096) - Decreases malignancy risk
5. **Clump Thickness** (β = -0.347 ± 0.055) - Decreases malignancy risk

### Algorithm Performance
- **Total annealing steps**: 8,248
- **Mean acceptance rate**: 0.75
- **Computation time**: ~13 minutes (784 seconds)
- **Final ESS/N**: 0.5

## Algorithm Details

The Annealed SMC algorithm:

1. Starts with particles sampled from the prior distribution
2. Gradually increases temperature parameter β from 0 to 1
3. At each step, reweights particles and performs MCMC moves
4. Resamples when effective sample size drops below threshold
5. Produces weighted samples from the posterior distribution

## Visualization

The analysis produces several diagnostic plots:

- Annealing schedule progression
- MCMC acceptance rates over iterations
- Posterior distributions for each coefficient
- Joint posterior for top features
- Particle weight distributions

## Future Work

- **Cross-validation**: Implement k-fold CV for out-of-sample evaluation
- **Model comparison**: Compare with simpler models using marginal likelihood
- **Feature selection**: Identify sparse models using posterior distributions
- **Sensitivity analysis**: Test robustness to different prior specifications

## References

- Del Moral, P., Doucet, A., & Jasra, A. (2006). Sequential Monte Carlo samplers. *Journal of the Royal Statistical Society: Series B*, 68(3), 411-436.
- Wisconsin Breast Cancer Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original))

## Author

Ravleen Bajaj

## License

MIT License
