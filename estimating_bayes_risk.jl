using Distributions
using QuadGK
using HCubature
using Plots
using LinearAlgebra

integrand(x, m) = abs( pdf(Normal(m, 1), x) - pdf(Normal(0,1), x))
bayes_risk(m) = 1/2 - (1/4)*( quadgk(x -> integrand(x, m), -Inf, Inf)[1] )
bayes_risk2(m) = 1 - cdf(Normal(0, 1), m/2)


m = 0:0.1:5
y = bayes_risk.(m)
y2 = bayes_risk2.(m)

plot(m, y)
plot!(m, y2)
plot!(m, y3)

# # # # # # # #

normal_pdf(μ) = x -> pdf(MvNormal(μ, 1*I), x)
integrand(m, d) = x -> abs( normal_pdf(zeros(d))(x) - normal_pdf( [m ; zeros(d-1)] )(x) )
calc_integral(m,d) = hcubature( integrand(m,d), fill(-4-m,d), fill(4+m,d))[1]
bayes_risk3(m,d) = 1/2 - (1/4)*(calc_integral(m,d))

y3 = (m -> bayes_risk3(m,3)).(m)
