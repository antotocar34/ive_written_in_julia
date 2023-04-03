using Random, Distributions, Plots, LaTeXStrings, LinearAlgebra, BenchmarkTools, InvertedIndices, DataFrames, CSV
using StaticArrays

Random.seed!(1234)

function mysample(n::Integer, d::Integer)
    U = Uniform(0,1)
    X = [SVector{d, Float64}(rand(U, d)) for i=1:n]
    y = Vector{Bool}(undef, n)
    @simd for i = 1:n
        @inbounds y[i] = rand(Bernoulli(mean(X[i])))
    end
    return (y, X)
end

function mynorm(A) 
  x = zero(eltype(A))
  for v in A
    x += v * v
  end
  sqrt(x)
end

function mydistmat(X)
  n = size(X)[1]
  out = zeros(n, n) # Initialise matrix of zeros
  for i in 1:n
    @inbounds out[i,i] = Inf
    for j in 1:(i-1)
        @inbounds out[j,i] = mynorm(X[i] - X[j])
    end
  end
  return Symmetric(out)
end


function nearest_neighbour(i::Int64,
                           M::Symmetric{Float64, Matrix{Float64}}, 
                           y::Vector{Bool},
                           k::Int64
                            )::Vector{Bool}
    distances = @view M[i,:] 
    y[sortperm(distances)][1:k]
end

function knn_classify(i::Int64, 
                      M::Symmetric{Float64, Matrix{Float64}}, 
                      y::Vector{Bool}, 
                      k::Int64 ):: Bool
    y_nearest = nearest_neighbour(i, M, y, k)
    mean(y_nearest) > 0.5
end

function calculate_error_probability(n::Int64,d::Int64,k::Int64):: Float64
    y, X = mysample(n,d)

    M = mydistmat(X)

    y_pred = [ knn_classify(i, M, y, k) for i=1:size(X)[1] ]
    mean(y_pred .!= y)
end


risk_sample = function(m::Int64,n::Int64,d::Int64,k::Int64)
    probabilities = [ calculate_error_probability(n,d,k) for i=1:m ]
    print(".")
    mean(probabilities) # Get risk
end

simulate = function()
    m = 500
    n_seq = 10:50:1000
    d_seq = [1,2,5,10,100]
    k_seq = [1,3,5,7,9]

    total = map(length, [n_seq, d_seq, k_seq]) |> prod
    print(total, " to do!\n")

    output = [(
               n=n,
               d=d,
               k=k,
               estimated_risk=risk_sample(m,n,d,k)
              ) 
              for n in n_seq 
              for d in d_seq 
              for k in k_seq
             ]
    return output
end

result_df = DataFrame(simulate())

CSV.write("./simulation_results.csv", result_df)
### 
f(d) = 1/2 - 1/6*(1/d)

p = plot(f, 0, 50, ylims=(0,1), xticks=0:5:50, legend=false, dpi=800)
xlabel!(L"Dimension: d")
ylabel!(L"R_{1NN}")

savefig(p, "q8_asymptotic_risk_plot.png")

