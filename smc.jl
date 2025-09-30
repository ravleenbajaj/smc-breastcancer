# smc_fixed.jl
module SMC
using Random, StatsBase, Distributions, LinearAlgebra

export annealed_smc

# Prior: Normal(0, prior_scale)
function log_prior(beta::Vector; prior_scale=3.0)
    return -0.5 * dot(beta, beta) / (prior_scale^2)
end

# Logistic log-likelihood (FIXED - numerically stable version)
function log_likelihood(beta::Vector, X::Matrix, y::Vector)
    η = X * beta
    # Use log-sum-exp trick for numerical stability
    return sum(@. y * η - log1p(exp(η)))
end

# Effective Sample Size
ess(weights) = 1 / sum(weights.^2)

# Resampling
function multinomial_resample(particles, weights)
    idx = sample(1:length(particles), Weights(weights), length(particles))
    return particles[idx, :], fill(1.0/length(particles), length(particles))
end

# MH mutation (FIXED - better proposal scaling)
function mh_mutation(particles, logtarget, X, y; nsteps=10, step_scale=0.1)
    N, d = size(particles)
    accept = 0
    
    # Scale step size by sqrt(d) for dimension-adaptive proposals
    adapted_scale = step_scale / sqrt(d)
    
    for i in 1:N
        β = copy(particles[i, :])
        logp = logtarget(β, X, y)
        
        for _ in 1:nsteps
            prop = β .+ adapted_scale .* randn(d)
            logp_prop = logtarget(prop, X, y)
            
            if log(rand()) < logp_prop - logp
                β, logp = prop, logp_prop
                accept += 1
            end
        end
        
        particles[i, :] = β
    end
    
    return particles, accept / (N*nsteps)
end

# Annealed SMC
function annealed_smc(X, y; N=500, mcmc_steps=10, step_scale=0.5, ess_frac=0.5, prior_scale=3.0)
    n, p = size(X)
    
    # Initialize particles from prior
    particles = randn(N, p) .* prior_scale
    weights = fill(1.0/N, N)
    
    β = 0.0
    betas = [β]
    
    ll_particles = [log_likelihood(particles[i, :], X, y) for i in 1:N]
    acc_hist = Float64[]
    
    while β < 1.0
        # Find next β adaptively
        function ess_at(bnew)
            w = log.(weights) .+ (bnew-β) .* ll_particles
            w .-= maximum(w)
            w = exp.(w)
            w ./= sum(w)
            return ess(w)
        end
        
        β_new = 1.0
        if ess_at(1.0) < ess_frac*N
            low, high = β, 1.0
            for _ in 1:20
                mid = (low+high)/2
                if ess_at(mid) < ess_frac*N
                    high = mid
                else
                    low = mid
                end
            end
            β_new = low
        end
        
        # Reweight
        inc = (β_new - β) .* ll_particles
        lw = log.(weights) .+ inc
        lw .-= maximum(lw)
        weights = exp.(lw)
        weights ./= sum(weights)
        
        β = β_new
        push!(betas, β)
        
        # Resample if necessary
        if ess(weights) < ess_frac*N
            particles, weights = multinomial_resample(particles, weights)
        end
        
        # Mutation
        logtarget(βv, X, y) = log_prior(βv; prior_scale=prior_scale) + β*log_likelihood(βv, X, y)
        particles, acc = mh_mutation(particles, logtarget, X, y, nsteps=mcmc_steps, step_scale=step_scale)
        push!(acc_hist, acc)
        
        # Update likelihoods
        ll_particles = [log_likelihood(particles[i, :], X, y) for i in 1:N]
    end
    
    return particles, weights, betas, acc_hist
end

end # module