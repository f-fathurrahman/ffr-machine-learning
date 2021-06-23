using Printf
import Statistics: mean
import StatsBase

import PyPlot
const plt = PyPlot
plt.rc("text", usetex=true)
plt.matplotlib.style.use("ggplot")

using Random
Random.seed!(1234)

function do_fit(x, t, Norder)
    @assert Norder >= 1
    Ndata = size(x,1)
    # Build X matrix
    X = zeros(Ndata,Norder+1)
    for i in 1:Ndata
        X[i,1] = 1.0
        for n in 1:Norder
            X[i,n+1] = x[i]^n
        end
    end
    # Calculate w
    w = inv(X' * X) * X' * t
    return w
end

function do_predict(w, x)
    Ndata = size(x,1)
    Norder = size(w,1) - 1
    t = ones(Ndata)*w[1]
    for n in 1:Norder
        t[:] = t[:] + w[n+1]*x.^n
    end
    return t
end

function generate_data_and_fit(w_true, σ2_true, Nsample)
    x = rand(Nsample)  # sample from uniform distribution
    # Generate
    t = zeros(Nsample)
    for i in 1:Nsample
        t[i] = w_true[1] + w_true[2]*x[i] + σ2_true*randn()
    end
    # fit
    w = do_fit(x, t, 1)
    return w[1], w[2]
end

function main()
    w_true = [-2.0, 3.0]
    σ2_true = 0.5^2
    Nsample = 20

    Ntrials = 10000
    w_0 = zeros(Ntrials)
    w_1 = zeros(Ntrials)
    for i in 1:Ntrials
        w_0[i], w_1[i] = generate_data_and_fit(w_true, σ2_true, Nsample)
    end
    
    #h = StatsBase.fit(StatsBase.Histogram, (w_0, w_1), nbins=10)

    plt.clf()
    plt.hist2D(w_0, w_1, bins=[30,30])
    plt.colorbar()
    plt.xlabel("\$w_0\$")
    plt.ylabel("\$w_1\$")
    plt.savefig("IMG_page78.pdf")

end

main()