using LinearAlgebra, BenchmarkTools, StaticArrays, Combinatorics

e(i, n) = begin 
        zero_v = zeros(Int8,n)
        zero_v[i] = 1
        return zero_v
       end

function generate_points(number_of_points, dimension)
    one_vector = ones(Int8, dimension)
    X = Vector{StaticVector{dimension, Int8}}(undef, number_of_points)
    for (i, x) in enumerate(multiset_permutations([one_vector ; -one_vector], dimension))
        if i > number_of_points break end
        @inbounds X[i] = SVector{dimension, Int8}(x)
    end
    return VectorOfArray(X)
end

X = generate_points(2, 4)

"""
Return true if there no halfspace contains the subset. 
"""
function not_separated(points)
    # Iterate through the dimensions
    shatterable = false
    for j=1:(size(points)[2]) # Iterate through vector coordinates
        # Check that all are of the same sign.
        @info points[:, j]
        bool = all( ==(x[1]), points[:, j]) 
        shatterable = bool
    end
    return shatterable
end

function main(number_of_points, dimension)
    @assert number_of_points <= 2^dimension
    X = generate_points(number_of_points, dimension)
    bool = true
    for subset in powerset(1:number_of_points, 1)
        bool = not_separated(X[subset])
    end
    return bool
end
