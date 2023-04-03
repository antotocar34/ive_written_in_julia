using Base: sort
using Distributions
using LinearAlgebra
using Plots
using StatsPlots
using KernelDensity
using Turing
using DataFrames


l1norm(v) = sum(abs(v_i) for v_i in v)

project = function(β, r)
    d = length(β)
    if l1norm(β) <= r
        return β
    end

    mysort(x::Number) = x
    mysort(x) = sort(x ; rev=true)

    β_sorted = mysort(abs.(β))

    μ(j) = max( sum(β_sorted[i] for i=1:j) - r,0)
    c = maximum([ j for j=1:d if β_sorted[j] > μ(j)/j ])

    μ_c = μ(c)

    θ = [ sign(β[i]) * max(abs(β[i]) - (μ_c/c), 0) for i=1:d ]
    if length(θ) == 1
        return θ[1]
    end
    return θ
end


function simulate_changepoints(; sigma=2)
    simulations = Vector{Float64}(undef, 50)

    mu_t = [30, 10, 40 , 20, 20]
    j = 1
    for t=1:50
        if mod(t, 20) == 0
            j += 1
        end
        simulations[t] = rand(Normal(mu_t[j], sigma))
    end
    return simulations
end

struct JeffreysPrior <: ContinuousUnivariateDistribution end
Distributions.logpdf(d::JeffreysPrior, x::Real) = -2*log(x) 

@model function changepoint(y, X)
    t = length(y)

    # λ ~ filldist(Exponential(1), t)
    λ ~ Exponential(1)
    β ~ arraydist([DoubleExponential(0, λ) for i=1:t])
    r ~ Exponential(1)
    σ2 ~ InverseGamma(1,1)

    θ = project(β, r)

    y ~ MultivariateNormal(X*θ, diagm(repeat([σ2], t)) )
end

y = simulate_changepoints()
t = length(y)
X = tril(ones(t, t))
out = sample(changepoint(y, X), NUTS(), 1000)
out_df = DataFrames.DataFrame(out)
betas = out_df[:, vcat([:r], [Symbol("β[$x]") for x in 1:50])]
betas_matrix = Matrix(betas)[1:100,:]

thetas = Matrix{Float64}(undef, 100, 50)
for i in 1:(size(betas_matrix)[1])
    β = betas_matrix[i, 2:end]
    r = betas_matrix[i, 1]
    thetas[i,:] = project(β, r) 
end

y_hat = mapslices(mean, thetas * X ; dims=[2])
plot(y,y_hat)
