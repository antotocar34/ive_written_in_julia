using Distributions
using Plots
using LinearAlgebra
using LaTeXStrings
using BenchmarkTools

do_plot = true
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

bellman_operator(policy, f) = [
                               r(x, policy(x)) + γ * sum( ( P(y, policy(x), x)*f[y+1] for y in X) )  
                               for x in X
                              ]

bellman_optimality_operator(f) = [
                                  maximum( 
                                      [r(x, a) + γ * sum( ( P(y, a, x)*f[y+1] for y in X)) for a in [high,low]]
                                     )
                                  for x in X
                                 ]

π_lazy(x) = low

π_agg(x) = x >= N/2 ? high : low

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
        P_π[i,j] = P(y, policy(x), x)
    end end
    return P_π
end

function solve_bellman_equation(policy)
    V_π = policy.(X)
    r_π = (x -> r(x, policy(x))).(X)
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
end

function policy_iteration(stop=Inf)
    V_0 = fill(2, N) # Start with arbitrary value function
    π_0 = x -> policy_V(V_0)
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

if do_plot
# testing that power_iteration and bellman_equation_solver are broadly similar.
V_pi = power_iteration(π_lazy)
V_bm = solve_bellman_equation(π_lazy)
@assert norm(V_pi - V_bm, Inf) < 1e-12

# Plotting difference between policy's value functions
pl = plot([solve_bellman_equation(π_lazy), solve_bellman_equation(π_agg)], labels=[L"V_{lazy}" L"V_{agg}"], dpi=600) ;
savefig(pl, "./plots/hw1/value_funcs.png" )
# Plotting difference between policy's value functions
pl = plot(solve_bellman_equation(π_lazy) .- solve_bellman_equation(π_agg), ylab=L"V_{lazy} - V_{agg}", legend=false, color="firebrick", dpi=600);
savefig(pl, "./plots/hw1/value_difference.png" )

# Plotting values after iteration
pl = plot( [ value_iteration(i).value_function for i in [10,20,50,100] ], labels=["10" "20" "50" "100"], title="Value iteration convergence", legendtitle="Iteration", legend=:bottomleft , dpi=600);
savefig(pl, "./plots/hw1/valueiteration_convergence.png" )
pl = plot( [ policy_iteration(i).value_function for i in [10,20,50,100] ], labels=["10" "20" "50" "100"], title="Policy iteration convergence", legendtitle="Iteration", legend=:bottomleft , dpi=600);
savefig(pl, "./plots/hw1/policyiteration_convergence.png" )

# Finding time differences
time_pi_100 = @belapsed policy_iteration(100)
time_vi_100 = @belapsed value_iteration(100)

time_pi = @belapsed policy_iteration()
time_vi = @belapsed value_iteration()

V_star = policy_iteration().value_function

# Plotting difference in value functions
V_lazy = solve_bellman_equation(π_lazy)
V_agg = solve_bellman_equation(π_agg)
pl = plot([V_star .- V_lazy, V_star .- V_agg], labels=[L"V^* - V_{lazy}" L"V^* - V_{agg}"], legend=:topleft, dpi=600)
savefig(pl, "./plots/hw1/optvaluefunc_diff.png" )

pl = plot( [Bool.(π_lazy.(X)) .+ 0.005, Bool.(π_agg.(X)) .- 0.005, Bool.(policy_V(V_star).(X))], legend=:topleft, grid=false, yticks=([0,1], ["low", "high"]), xticks=0:5:N, labels=["Lazy" "Aggressive" "Optimal"], dpi=600)
savefig(pl, "./plots/hw1/optimal_policy.png")

end
