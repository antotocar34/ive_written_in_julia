using Plots
using LinearAlgebra
using LaTeXStrings
using DataFrames
using BenchmarkTools
using SparseArrays

const run_simulations = false
# CONSTANTS
const N = 100 # Maximum size of queue
const X = 0:(N-1) # State space
const γ = 0.9 # Gamma parameter
const p = 0.5 # Probability of queue increasing

struct Action
    high::Bool
end

Base.Bool(a::Action) = a.high

const high = Action(true)
const low = Action(false)

r(x,a::Action) = -(x/N)^2 - c(a) # reward function

c(a::Action) = a.high ? 0.01 : 0 # cost function

q(a::Action) = a.high ? 0.6 : 0.51

transition = function(a::Action, x_prev) # Transition function
    u = rand()
    x_new = x_prev + (u < p) - (u < q(a))
    return min(N-1, max(x_new, 0))
end

# Transition probability function
P = function(x, a::Action, x_prev)
    @assert (x in X) && (x_prev in X)
    # Corner cases
    if x == x_prev == N-1
        # Cases where you stay in place
        return p*(1-q(a)) + p*q(a) + (1-p)*(1-q(a))
    end

    if x == x_prev == 0
        return q(a)*(1-p) + p*q(a) +  (1-q(a))*(1-p)
    end

    # General case
    if abs(x - x_prev) > 1
        return 0
    elseif x == (x_prev + 1)
        return p * (1 - q(a))
    elseif x == (x_prev - 1)
        return (1 - p) * q(a)
    else
        return p*q(a) + (1-p)*(1-q(a))
    end
end

bellman_operator(policy::Vector, f) = [
                               r(x, policy[x+1]) + γ * sum( ( P(y, policy[x+1], x)*f[y+1] for y in X) )  
                               for x in X
                              ]

bellman_optimality_operator(f) = [
                                  maximum( 
                                      [r(x, a) + γ * sum( ( P(y, a, x)*f[y+1] for y in X)) for a in [high,low]]
                                     )
                                  for x in X
                                 ]

const π_lazy = map(x->low, X)

const π_agg = map(x -> x >= N/2 ? high : low, X)

function power_iteration(policy)
    V_0 = fill(1, N) # Start with an arbitrary value function
    V_k_plusone = V_0 # Initialise
    converged = false
    while !converged
        V_k = V_k_plusone
        V_k_plusone = bellman_operator(policy, V_k)
        converged = norm(V_k_plusone - V_k, Inf) < 1e-20 ? true : false 
    end
    return V_k_plusone
end

make_P_matrix = function(policy)
    P_π = Matrix{Float64}(undef, length(X), length(X))
    for (i,x) in enumerate(X) for (j,y) in enumerate(X)
        P_π[i,j] = P(y, policy[x+1], x)
    end end
    return P_π
end

function solve_bellman_equation(policy::Vector)
    V_π = policy
    r_π = map(x -> r(x, policy[x+1]), X)
    P_π = make_P_matrix(policy)
    return inv(I - γ*P_π) * r_π
end

## Optimal policy calculation
function value_iteration(stop=Inf)
    V_0 = fill(2, N) # Start with the 0 value function
    V_k_plusone = V_0 # Initialise
    converged = false
    i = 0
    while !converged
        V_k = V_k_plusone
        V_k_plusone = bellman_optimality_operator(V_k)
        converged = norm(V_k_plusone - V_k, Inf) < 1e-10 ? true : false 

        if i == stop
            break
        end
        i += 1
    end
    return (value_function=V_k_plusone, iterations=i)
end

function policy_V(V)
    mapping = x -> [high,low][argmax([ 
                              r(x, a) + γ*sum((P(y,a,x)*V[y+1] for y in X))
                              for a in [high, low]
                                    ])]
    map(mapping, X)
end

function policy_iteration(stop=Inf)
    V_0 = fill(2, N) # Start with arbitrary value function
    π_0 = policy_V(V_0)
    π_k_plusone = π_0 # Initialise policy
    V_k_plusone = V_0 # Initialise Value function
    converged = false
    i = 0
    while !converged
        V_k = V_k_plusone
        π_k = policy_V(V_k)

        V_k_plusone = solve_bellman_equation(π_k)
        converged = norm(V_k - V_k_plusone, Inf) < 1e-10 ? true : false

        if i == stop
            break
        end
        i += 1
    end
    return (policy=policy_V(V_k_plusone), value_function=V_k_plusone, iterations=i)
end

## HW2.jl
# Defining feature vectors and precomputing for speed
ϕ_fine_ = function(x::Integer)
    e_i = zeros(N)
    e_i[x+1] = true # Add 1 to account for 1 based indexing 
    return e_i
end
const ϕ_fine = map(ϕ_fine_, X)

ϕ_coarse_(x::Integer) = [ (5(i-1)<= x <= 5i-1) for i=1:(N)/5]
const ϕ_coarse = map(ϕ_coarse_, X)

ϕ_pwl_ = function(x::Integer)
    indicator(x, i) = (5(i-1)<= x <= 5i-1) ? 1 : 0
    return vcat( ϕ_coarse[x+1], [ indicator(x,i) * (x-5(i-1))/5 for i=0:N/5 ])
end
const ϕ_pwl = map(ϕ_pwl_, X)

sample_mdp = function(policy, x_0, number_of_samples)
    number_of_samples = Int64(number_of_samples)
    history = Dict( 0 => (state=x_0, action=policy[x_0+1]) )
    for t=1:number_of_samples
        state = transition(history[t-1].action, history[t-1].state)
        action = policy[state+1]
        history[t] = (state=state, action=action)
    end
    return history
end

sample_mdp_v2 = function(policy, x_0, number_of_samples)
    number_of_samples = Int64(number_of_samples)
    history = Dict( 0 => (state=x_0, action=policy[x_0+1]) )
    for t=1:number_of_samples
        state = transition(history[t-1].action, history[t-1].state)
        action = policy[state+1]
        history[t] = (state=state, action=action)
        if (history[ t>10 ? (t-10) : 0].state == 0)
            history[t] = (state=x_0, action=action)
        end
    end
    return history
end

TD_0 = function(policy, ϕ ; number_of_samples=1e4, x_0=N-1, max_iterations=1e7, alternate_sampling=false)
    if alternate_sampling
        history = sample_mdp_v2(policy, x_0, number_of_samples)
    else
        history = sample_mdp(policy, x_0, number_of_samples)
    end

    a = 1e5 ; b = 1e5
    α(t) = a / (t+b)

    θ_t = zeros( length(ϕ[1]) )
    t = 0
    @inbounds for t=1:max_iterations
        x_t = history[mod(t, number_of_samples+1)].state
        x_tp1 = history[mod(t+1, number_of_samples+1)].state
        δ_t = r(x_t, policy[x_t+1]) + (γ * θ_t'ϕ[x_tp1+1]) - θ_t'ϕ[x_t+1]
        θ_t = θ_t + (α(t)*δ_t*ϕ[x_t+1])
        t += 1
        # if mod(t, 1000000) == 0
        #     println(t)
        # end
    end
    return map(x -> ϕ[x+1]'θ_t, X)
end

LSTD = function(policy::Vector, ϕ; x_0=N-1, number_of_samples=1e5, alternate_sampling=false)
    d = length(ϕ[3])

    if alternate_sampling
        history = sample_mdp_v2(policy, x_0, number_of_samples)
    else
        history = sample_mdp(policy, x_0, number_of_samples)
    end

    M = begin
        out = zeros(Float64, d, d)
        @inbounds for t=1:(number_of_samples-1)
            x_t = history[t].state
            x_tp1 = history[t+1].state
            out += ϕ[x_t+1] * ϕ[x_t+1]' - γ*(ϕ[x_t+1] * ϕ[x_tp1+1]')
        end
        out
    end

    b_T = begin
        out = zeros(d)
        @inbounds for t=0:(number_of_samples-1)
            x_t = history[t].state
            a_t = history[t].action
            out += r(x_t, a_t) * ϕ[x_t+1]
        end
        out
    end

    θ_T = (M + 1e-12*I) \ b_T
    @assert !any(isnan.(θ_T)) "\nθ_T:$θ_T"
    return map(x -> ϕ[x+1]'θ_T, X)
end

LSTD_faster = function(policy:: Vector, ϕ ; x_0=N-1, number_of_samples=1e5, alternate_sampling=false)
    #=
    Algorithm in section 9.8 of
    `Reinforcement Learning`
    by Sutton and Barto
    =#
    d = length(ϕ[1])
    if alternate_sampling
        history = sample_mdp_v2(policy, x_0, number_of_samples)
    else
        history = sample_mdp(policy, x_0, number_of_samples)
    end

    ϵ_inv = 1e-9
    A_hat = (1/ϵ_inv)*diagm(ones(d))
    b_hat = zeros(d)
    w = zeros(d)
    for t=1:(number_of_samples-1)
        x_t = history[t].state
        a_t = history[t].action
        x_tp1 = history[t+1].state
        v = A_hat' * (ϕ[x_t+1] - γ*ϕ[x_tp1+1])
        A_hat = A_hat - (1/(1+(v'ϕ[x_t+1])))*( (A_hat*ϕ[x_t+1])*v' )
        b_hat = b_hat + r(x_t, a_t) * ϕ[x_t+1]
        w = A_hat * b_hat
    end
    return map(x -> ϕ[x+1]'w, X)
end

function q1_simulate(alternate_sampling)
    ϕs = Dict(
              "Fine" => ϕ_fine,
              "Coarse" => ϕ_coarse,
              "Piecewise Linear" => ϕ_pwl
             )
    policies = Dict(
              "Lazy" => π_lazy,
              "Aggressive" => π_agg
                   )
    sample_nums = Int64.([1e4, 1e5, 1e6, 1e7])

    for policy in keys(policies) for ϕ in keys(ϕs)
        for number_of_samples in sample_nums
            println("Calculating LSTD")
            V_LSTD = LSTD_faster(policies[policy], ϕs[ϕ] ; number_of_samples=number_of_samples)
            println("Calculating TD(0)")
            V_TD = TD_0(policies[policy], ϕs[ϕ] ; number_of_samples=number_of_samples)
            V_π = solve_bellman_equation(policies[policy])

            legend = round(log(10,number_of_samples)) == 6 ? :bottomleft : false

            pl1 = plot(hcat(V_π, V_TD, V_LSTD), dpi=400, labels=["$policy Value function" "TD(0)" "LSTD"], 
                       # title="$policy VF with $ϕ encoding and 10^$(Int64(round(log(10,number_of_samples)))) samples",
                       title="10^$(Int64(round(log(10,number_of_samples)))) Samples",
                       titlefont=font(12, "Computer Modern"),
                       legend=legend)
            text = alternate_sampling? "_as_" : ""
            savefig(pl1, "../code/plots/hw2/policy$(policy)_ϕ$(ϕ)_number_of_samples$(number_of_samples)"*text*".png")
        end 
    end end
end
if run_simulations
    q1_simulate(false)
    q1_simulate(true)
end

function evaluator(V::Vector, x)
    #=
    Helper function to evaluate vector index,
    which takes zero if this index is out of bounds
    by 1
    =#
    if x in [0, length(V)+1]
        return 0
    else 
        return V[x] 
    end
end

function approx_policy_iteration(ϕ ; iterations=100, number_of_samples=1e6 alternate_sampling=false)
    π_k = π_lazy # Initialize policy
    V_k = map(x->0, X)
    Q_k(x, a::Action , V_k::Vector) = begin
        (
         r(x,a) + 
         γ*(1-p)*(q(a)*evaluator(V_k, x) + (1-q(a))*V_k[x+1]) + 
         γ*p*(q(a)*V_k[x+1] + (1-q(a))*evaluator(V_k, x+2))
        )
        end
    for k=1:iterations
        print(k, " ")
        V_k = LSTD_faster(π_k, ϕ ; number_of_samples=number_of_samples, alternate_sampling=alternate_sampling)
        π_k = [
               [high, low][argmax(
                                    [
                                     Q_k(x,a,V_k) for a in [high, low]
                                    ]
                                   )]
               for x in X]
    end
    return (V_k, π_k)
end

function q2_simulate(alternate_sampling=false)
    ϕs = Dict(
              "Fine" => ϕ_fine,
              "Coarse" => ϕ_coarse,
              "Piecewise Linear" => ϕ_pwl
             )
    iterations_list = [10, 100]
    n = length(ϕs)*length(iterations_list)
    result = DataFrame(Dict(
                            :encoding => Vector{String}(undef, n),
                            :iterations => Vector{Integer}(undef, n),
                            :V_appr => Vector{Vector}(undef, n),
                            :policy_appr => Vector{Vector}(undef, n)
                           )
                      )

    i = 1
    for ϕ in keys(ϕs) for iteration in iterations_list
        ϕ_func = ϕs[ϕ]
        V_pi, policy_appr = approx_policy_iteration(ϕ_func ; iterations=iteration, number_of_samples=1e5, alternate_sampling) 

        result[i, :] = Dict(
                            :encoding    => ϕ,
                            :iterations  => iteration,
                            :V_appr      => V_pi,
                            :policy_appr => policy_appr
                           )
        i += 1
    end end
    return result
end
const result_df = q2_simulate()
const result_df_2 = q2_simulate(true)


function make_plot(encoding, result_df, alternate_sampling=false)
    subsetted = filter(
                       :encoding => x -> (x==encoding)
                     , result_df)
    V_true = value_iteration().value_function
    val_funcs_plot = plot(
                      [V_true, 
                       subsetted.V_appr[1],
                       subsetted.V_appr[2]],
                      legend=false,
                      title="$encoding Encoding",
                      titlefont=font(12, "Computer Modern")
                        );
    policy_plot = plot(
                    [
                     Bool.(policy_V(V_true)) .+ 0.05,
                     Bool.(subsetted.policy_appr[1]) .+ 0.025,
                     Bool.(subsetted.policy_appr[2])
                    ],
                     yticks=([0,1], ["low", "high"]), 
                     xticks=0:5:N,
                     grid=false,
                     legend=:topleft,
                     labels=[ "True" "10 iterations" "100 iterations"]
                      );
    pl = plot(val_funcs_plot, policy_plot, dpi=400, layout=(2,1))
    text = alternate_sampling? "_as_" : ""
    savefig(pl, "../code/plots/hw2/Q2_$encoding"*text*".png")
end

function make_plots()
    for encoding in ["Fine", "Coarse", "Piecewise Linear"]
        make_plot(encoding, result_df)
        make_plot(encoding, result_df_2)
    end
end

make_plots()
