# main.jl
include("dataset.jl")
include("smc.jl")
using .SMC

# Load data
X, y = load_breast_cancer()
println("Dataset size: ", size(X))

# Run Annealed SMC
particles, weights, betas, acc = SMC.annealed_smc(X, y; N=300, mcmc_steps=5)

# Posterior mean
posterior_mean = vec(sum(particles .* weights, dims=1))
println("Posterior mean estimate: ", posterior_mean)
println("Annealing schedule: ", betas)
