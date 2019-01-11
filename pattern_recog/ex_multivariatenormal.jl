using Distributions

Sigma = [2.0 0.0
         0.0 1.0]
mu = [1.0, 0.1]
A = rand(MultivariateNormal(mu, Sigma), 10)
println(A)

