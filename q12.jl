using Plots


f(k,d) = min( 2k * log2( (exp(1)*d)/k), d)

d = 50
p = plot(x -> f(x,d), 0, 100, xticks=0:5:100, dpi=500, label="VC-dimension lower bound")

savefig("../figures/lowerbound.png")
