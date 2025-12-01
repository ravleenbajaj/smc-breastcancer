# Annealed Sequential Monte Carlo for Bayesian Logistic Regression

A Julia implementation of Annealed Sequential Monte Carlo (SMC) for Bayesian inference on logistic regression, applied to the Wisconsin Breast Cancer dataset.

## Overview

This project is a simple example that demonstrates how to use Sequential Monte Carlo with annealing to sample from posterior distributions of logistic regression coefficients. The algorithm gradually transitions from the prior distribution (β=0) to the posterior distribution (β=1), making it particularly effective for complex, multimodal posteriors.

- **Annealed SMC Implementation**: Adaptive temperature scheduling to maintain effective sample size
- **Bayesian Logistic Regression**: Full posterior inference for classification problems
- **Feature Analysis**: Identification of statistically significant predictors with credible intervals
- **Model Diagnostics**: Acceptance rates, effective sample size tracking, and weight distributions

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
import Pkg
Pkg.add(["CSV", "DataFrames", "Printf", "Plots", "StatsBase", "HTTP", "Distributions"])
```

## Project Structure

```
smc-breastcancer/
├── analysis.jl          # Main analysis script
├── dataset.jl           # Data loading utilities
├── smc.jl              # Annealed SMC implementation
├── README.md           # This file
└── Project.toml        # Julia project dependencies
```

## Usage

```julia
# Load required modules
Using Random
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
prior_scale =  3.0

# Run annealed SMC
particles, particle_weights, betas, acc_hist = SMC.annealed_smc(
    X, y;
    N=N_particles,
    mcmc_steps=mcmc_steps,
    step_scale=step_scale,
    ess_frac=ess_threshold,
    prior_scale=prior_scale
)
```

## Results

### Model Performance
- **Accuracy**: 86.55%
- **Sensitivity**: 0.842
- **Specificity**: 0.878
- **Precision**: 0.784

### Top Predictive Features

Features with 95% credible intervals excluding zero:

1. **Cell Size** (β = 0.932 ± 0.131) - Increases malignancy risk
2. **Epithelial Cell Size** (β = -0.829 ± 0.100) - Decreases malignancy risk
3. **Bare Nuclei** (β = 0.570 ± 0.059) - Increases malignancy risk
4. **Bland Chromatin** (β = -0.522 ± 0.089) - Decreases malignancy risk
5. **Clump Thickness** (β = -0.340 ± 0.060) - Decreases malignancy risk

### Performance
- **Total annealing steps**: 8,504
- **Mean acceptance rate**: 0.323
- **Computation time**: ~10 minutes (606 seconds)
- **Final ESS/N**: 0.5

## Algorithm Details

The Annealed SMC algorithm:

1. Starts with particles sampled from the prior distribution
2. Gradually increases temperature parameter β from 0 to 1
3. At each step, reweights particles and performs MCMC moves
4. Resamples when effective sample size drops below threshold
5. Produces weighted samples from the posterior distribution

The algorithm uses a sequence of intermediate distributions to smoothly transition from prior to posterior:
π_β(θ) ∝ π_0(θ)^(1-β) × π_1(θ)^β
where:

β ∈ [0, 1] is the annealing parameter
π_0(θ) is the prior distribution
π_1(θ) is the posterior distribution

The temperature schedule is adaptively chosen to maintain effective sample size above the threshold, triggering resampling when particle degeneracy occurs.


## Future Goals

- **Interaction plots** for selection probabilities.
- **Parallel tempering** for improved mixing.
- **Model comparison**, i.e., compare with simpler models using marginal likelihood
- **Feature selection**, i.e., identify sparse models using posterior distributions



## References

- Del Moral, P., Doucet, A., & Jasra, A. (2006). Sequential Monte Carlo samplers. *Journal of the Royal Statistical Society: Series B*, 68(3), 411-436.
- Wisconsin Breast Cancer Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original))

## Author

Ravleen Bajaj

## License

MIT License
