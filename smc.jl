# smc.jl
module SMC

using Random, StatsBase, Distributions, LinearAlgebra

# Prior: standard normal
function log_prior(beta::Vector)
    return -0.5 * dot(beta, beta)
end

# Logistic log-likelihood
function log_likelihood(beta::Vector, X::Matrix, y::Vector)
    η = X * beta
    return sum(y .* log.(1 ./(1 .+ exp.(-η))) .+ (1 .- y) .* log.(1 ./(1 .+ exp.(η))))
end

# Effective Sample Size
ess(weights) = 1 / sum(weights.^2)

# Resampling
function multinomial_resample(particles, weights)
    idx = sample(1:length(particles), Weights(weights), length(particles))
    return particles[idx], fill(1.0/length(particles), length(particles))
end

# MH mutation
function mh_mutation(particles, logtarget, X, y; nsteps=3, step_scale=0.1)
    N, d = size(particles)
    accept = 0
    for i in 1:N
        β = copy(particles[i, :])
        logp = logtarget(β, X, y)
        for _ in 1:nsteps
            prop = β .+ step_scale .* randn(d)
            logp_prop = logtarget(prop, X, y)
            if log(rand()) < logp_prop - logp
                β, logp = prop, logp_prop
                accept += 1
            end
        end
        particles[i, :] .= β
    end
    return particles, accept / (N*nsteps)
end

# Annealed SMC
function annealed_smc(X, y; N=200, mcmc_steps=3, step_scale=0.05, ess_frac=0.5)
    n, p = size(X)
    particles = randn(N, p)
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
        logtarget(βv, X, y) = log_prior(βv) + β*log_likelihood(βv, X, y)
        particles, acc = mh_mutation(particles, logtarget, X, y, nsteps=mcmc_steps, step_scale=step_scale)
        acc_hist = push!(acc_hist, acc)
        ll_particles = [log_likelihood(particles[i, :], X, y) for i in 1:N]
    end
    return particles, weights, betas, acc_hist
end

end # module
