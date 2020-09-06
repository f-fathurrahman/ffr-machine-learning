using Printf
import Statistics: mean

import PyPlot
const plt = PyPlot
plt.rc("text", usetex=true)
plt.matplotlib.style.use("ggplot")

using Random
Random.seed!(1234)

function main()

    w_true = [-2.0, 3.0]
    σ2_true = 0.5^2

    Nsizes = range(20, stop=1000, step=20)

    Nsample = 10000 # Number of datasets to be averaged over

    all_ss = zeros(length(Nsizes),Nsample)

    for j in 1:length(Nsizes)
        Ndata = Nsizes[j]   # Number of objects        
        x = rand(Ndata)     # sample random data, from uniform distrib
        X = zeros(Ndata,2) # Matrix for evaluating input
        for i in 1:Ndata
            X[i,1] = 1.0
            X[i,2] = x[i]
        end
        # Generate different data set (different random number)
        for i in 1:Nsample
            t = X*w_true + randn(Ndata)*sqrt(σ2_true)
            w = inv(X'*X) * X' * t
            σ2 = (t'*t - t'*X*w)/Ndata # estimate σ2 from data
            all_ss[j,i] = σ2
        end
        @printf("j = %d is done\n", j)
    end

    plt.clf()
    plt.plot(Nsizes, mean(all_ss,dims=2), label="empirical")
    plt.plot(Nsizes, σ2_true*(1.0 .- 2.0./Nsizes), label="theoretical") # expectation value
    plt.legend()
    plt.grid(true)
    plt.savefig("IMG_w_variation.pdf")

end

main()