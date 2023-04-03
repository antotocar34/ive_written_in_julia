using Distributions, LinearAlgebra
using DataFrames, DataFramesMeta, CSV
using StaticArrays
using ProgressMeter
using BenchmarkTools

sample_data = function(n, d)
    X = rand(MvNormal(zeros(d), Diagonal([1/s for s=1:d])), n)' # Return an n x d matrix
    y = X*ones(d) + rand(MvNormal(zeros(n), 1*I))
    return X, y
end

ols(X, y) = X \ y

function gd(X, y ; η = (t-> 1/√t) )
    n = size(X, 1)
    ws = Vector{Vector{Float64}}(undef, n+1)
    ws[1] = zeros(size(X, 2))
    for t=2:n+1
        ws[t] = ws[t-1] - η(t) * (2/n) * sum( ((ws[t-1]'X[i,:] - y[i])*X[i,:] for i=1:n) )
    end
    return mean(ws[2:n+1])
end

function sgd(X, y ; η = (t->1/√t))
    n = size(X, 1)
    w_minusone = - 2 * η(1) * y[1]*X[1,:]
    acc_w = w_minusone
    for t=2:n
        w_t = w_minusone - (2 * η(t) * (w_minusone'X[t,:] - y[t])*X[t,:])
        acc_w += w_t
        w_minusone = w_t
    end
    return acc_w / n
end

function calc_risk(estimator, m, n, d ; η=false, η_func=nothing)
    risk_acc = 0
    X_0, y_0 = sample_data(n, d)
    w = η ? estimator(X_0, y_0 ; η = η_func) : estimator(X_0, y_0)
    for _=1:m
        n_test = 1000
        X, y = sample_data(n_test, d)
        risk_acc += (y - (X*w)).^2 |> mean
    end
    return risk_acc / m
end

simulate = function()
    m = 100
    n_seq = 10:50:1000
    d_seq = [[1] ; 10:20:110]
    c_seq = [0, 0.001, 0.01, 0.1, 0.5, 1, 10]
    methods = Dict(
                   "ols" => ols,
                   "gd"  => gd,
                   "sgd" => sgd
                  )

    total = prod( length(n_seq) * length(d_seq) ) * (1 + 2*(length(c_seq)) )

    # Initialize array
    output = Array{
                   typeof(
                     (n=1, d=1, c=0.1, risk=0.1, method="string")
                    )
                  }(undef, total)

     i = 1
     @showprogress for n in n_seq; for d in d_seq; for method in keys(methods);
         if method == "ols"
             risk = calc_risk(methods["ols"], m, n, d)
             output[i] = (n=n, d=d, c=0, risk=risk, method="ols")
             i += 1
         else
             for c in c_seq
                 risk = calc_risk(methods[method], m, n, d
                                  ; η=true, η_func = ifelse(c==0, (t->1/√t),(t->c))
                                 )
                 output[i] = (n=n, d=d, c=c, risk=risk, method=method) 
                 i +=1
             end
         end
     end end end

     return output
end

result_df = DataFrame(simulate())
result_df.risk = map( x -> isfinite(x) ? x : NaN, result_df.risk)

CSV.write("./simulation_results.csv", result_df)

timing = function()
    algorithms = [sgd, ols, gd]
    n_seq = [10, 100, 1000, 10000]
    d_seq = [1, 10, 100, 1000]
    total = length( n_seq ) * length( d_seq  ) * length(algorithms)

    results = Vector{typeof((n=2, d=2, algorithm="", time=0.0))}(undef, total)

    [algo(sample_data(10,10)...) for algo in algorithms] # Run so that they are precompiled

    i = 1
    @showprogress for n in n_seq; for d in d_seq;
        for algorithm in algorithms
            println("Running $algorithm, n: $n, d: $d")
            X, y =  sample_data(n, d)
            time_elapsed = @elapsed algorithm(X,y)
            results[i] = (n=n, d=d, algorithm=String(nameof(algorithm)), time=time_elapsed)
            i += 1
        end
    end end
    return results
end

timing_results = timing() |> DataFrame

CSV.write("./timing_results.csv", timing_results)
