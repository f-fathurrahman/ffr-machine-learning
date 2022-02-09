using Distributions
using LinearAlgebra

using Random
Random.seed!(1234)

import PyPlot
const plt = PyPlot
plt.rc("text", usetex=true)
plt.matplotlib.style.use("ggplot")

function main()
    #Σ = [3.0 2.0; 2.0 3.0]
    Σ = [3.0 10.0; 0.0 10.0]
    μ = [-3.0, 6.0]
    Σ = 0.5*(Σ + Σ')
    
    λ, v = eigen(Σ)
    #
    println("eigval 1 of Σ: ", λ[1])
    println("eigvec 1 of Σ: ")
    display(v[:,1]); println()
    #
    println("eigval 1 of Σ: ", λ[2])
    println("eigvec 2 of Σ:")
    display(v[:,2]); println()

    rndgen = MvNormal(μ, Σ)
    Nsamples = 1000

    data = rand(rndgen, Nsamples)
    plt.clf()
    plt.scatter(data[1,:], data[2,:])
    plt.axis("equal")
    plt.savefig("IMG_sample_mvnormal.pdf")
end

main()