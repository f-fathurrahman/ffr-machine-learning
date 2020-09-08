using Distributions
using LinearAlgebra

using Random
Random.seed!(1234)

import PyPlot
const plt = PyPlot
plt.rc("text", usetex=true)
plt.matplotlib.style.use("ggplot")

function main()
    σ_1 = 0.5
    μ_1 = 1.1

    σ_2 = 1.5
    μ_2 = -1.0

    Nsamples = 10000
    
    data1 = μ_1 .+ σ_1*randn(Nsamples)
    data2 = μ_2 .+ σ_2*randn(Nsamples)

    plt.clf()
    plt.hist(data1, bins=40, label="data1", alpha=0.8, edgecolor="None")
    plt.hist(data2, bins=40, label="data2", alpha=0.8, edgecolor="None")
    plt.legend()
    plt.savefig("IMG_sample_hist_normal.pdf")
end

main()