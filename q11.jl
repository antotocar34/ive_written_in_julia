using Distributions, StaticArrays, Parameters, LinearAlgebra, Gadfly
using DataFrames, CSV, ProgressMeter
#using GLMakie
import Cairo, Fontconfig

data_sample_separable = function(n, d, a)
    U_a = Uniform(-1, -a)
    U_b = Uniform(a, 1)
    U = Uniform(-1, 1)
    y = rand( [1,-1], n)
    X = Vector{Vector{Float64}}(undef, n)
    #@inbounds for i=1:n
    for i=1:n
        first_val = y[i]==1 ? rand(U_a) : rand(U_b)
        x = Vector{Float64}(undef, d)
        x[1] = first_val
        for j=2:d
            x[j] = rand(U) 
        end
        X[i] = x
    end
    return (y, X)
end

data_sample_nonseparable = function(n,d,m)
    y = rand( [1,-1], n)
    m_mean = [[m] ; zeros(d-1)]
    # X = @inbounds [
    X = [
         SVector{d,Float64}( 
                rand( 
                     MvNormal( 
                               ifelse(y[i]==1, zeros(d), m_mean) , 1*I 
                             ) 
                    )
               )
         for i=1:n
        ]
    return (y, X)
end

sample_data(n, d, parameter ; separable) = separable ? data_sample_separable(n,d,parameter) : data_sample_nonseparable(n,d,parameter) 

function perceptron(y, X ; max_iterations=500, verbose=false)
    d = size(X[1])[1]
    n = length(y)
    w_0 = zeros(d) # Initialize empty vector
    w_t_minus_1 = w_0
    w_t = w_0

    w_best = w_0

    # @inbounds for j=1:max_iterations
    for j=1:max_iterations
        for i=1:n
            if sign(w_t'X[i]) != y[i]
               new_w_t = w_t + y[i]*X[i]
               w_t_minus_1 = w_t
               w_t = new_w_t
           end
        end

        if w_best == w_t
            if verbose print("Algorithm has converged in ", j,  " steps!\n") end
            return (w_star=w_t, iterations=j)
        else
            w_best = w_t
        end
    end
    if verbose print("Algorithm did not converge!\n") end
    return (w_star=w_t, iterations=max_iterations)
end

function estimate_error_prob(w, d, parameter, separable)::Float64
    classifier(x, w) = w'x >= 0 ? 1 : -1

    m = 100
    n_test = 100

    probs = Vector{Float64}(undef, m)

    # @inbounds for j=1:m
    for j=1:m
        y, X = sample_data(n_test, d, parameter ; separable)

        y_pred = [classifier(X[i],w) for i=1:(size(X, 1))] # Calculate x'w for every row vector of X
        probs[j] = mean(y .!= y_pred)
    end
    return mean(probs) # Return monte carlo estimate of probability

end

function estimate_perceptron(n,d,a,m,separable; max_iterations=500)
    parameter = separable ? a : m
    y, X = sample_data(n, d, parameter ; separable = separable)

    out = perceptron(y, X, ; max_iterations=max_iterations, verbose=false)

    # Generate a new dataset to test.
    error_prob = estimate_error_prob(out.w_star, d, parameter, separable)

    return (n=n, d=d, a=a, m=m, 
            separable=separable, 
            iterations=out.iterations,
            error_prob=error_prob)
end

function simulate()
   n_seq = 10:50:1001
   d_seq = 1:10:100
   m_seq = 0.1:1:10.1
   a_seq = 0:0.05:0.99
   separable_seq = false:true

   total = prod((length(seq) for seq in [n_seq, d_seq])) * ( length(a_seq) + length(m_seq) )

   # Initialize array
   output = Array{
                  typeof((n=1, d=1, a=1.0, m=1.0, separable=true, iterations=1, error_prob=0.))
                 }(undef, total)

   i = 1
   @showprogress for n in n_seq; for d in d_seq; 
       for bool in separable_seq
           if bool
               for a in a_seq
                   output[i] = estimate_perceptron(n, d, a, 0., bool)
                   i += 1
               end
           else
               for m in m_seq
                   output[i] = estimate_perceptron(n, d, 0., m, bool)
                   i += 1
               end
           end
       end
   end end

   return output
end

result_df = DataFrame(simulate())


CSV.write("simulation_results.csv", result_df)

## PLOTTING

y, X = data_sample_nonseparable(100, 2, 20)

df = DataFrame()

function populate_df!(df::DataFrame, y::Vector{Int64}, X)
    df[!, :y] = y
    for j in 1:length(X[1])
        symb = Symbol("x$j")
        @inbounds df[!, symb] = [X[i][j] for i=1:length(X)]
    end
end

w_star = perceptron(y, X)
populate_df!(df, y, X)
y_range = -1:0.01:1
hyperplane_f(x) = -(w_star[2]/w_star[1])*x

points = layer(df, x=:x1, y=:x2, color=:y, Geom.point);
hyperplane = layer(x=hyperplane_f.(y_range), 
                   y=y_range, 
                   Geom.line,
                   color=[colorant"black"]);
p = Gadfly.plot( hyperplane,
          points,
          Scale.color_discrete(),
          Coord.cartesian(xmin=-1, xmax=1, ymin=-1, ymax=1)
         );
draw(PDF("test.pdf"),p)


# y, X = data_sample_separable(1000, 3, 0.01)

# w_star = perceptron(y, X, 10000)

# simulate()
