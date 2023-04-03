using Distributions, LinearAlgebra


functions = Dict(
                 "exponential" => (func=exp, derivative=exp),
                 "log_exponential" => (func=(x -> log2(1 + exp(x))), derivative=(x -> ))
                )


data_sample = function(n,d,a)
    y = rand(Bernoulli(1/2), n)
    X = Vector{Vector{Float64}}(undef, n)
    for i=1:n
        x = rand(MvNormal([ (y[i]==1 ? a : -a); zeros(d-1)] , 1*I))
        @inbounds X[i] = x
    end
    return (y=y, X=X)
end

SGD = function(y, X, phi)
    n = size(X)[1]
    func = w -> (1/n) * sum(phi(w'()))
end

train_classifier = function(y, X, phi)
    minimum = SGD(y, X, phi)
end



