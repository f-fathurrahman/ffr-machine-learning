using Printf
import Statistics: mean

import PyPlot
const plt = PyPlot
plt.rc("text", usetex=true)
plt.matplotlib.style.use("ggplot")

using Random

function main(Norder)
    Random.seed!(1234)

    N = 100 # Number of training points

    x = sort( 10.0*rand(N) .- 5 )
    t = 5*x.^3 .- x.^2 .+ x # true model
    σ2_true = 300.0         # true noise parameter
    t = t + randn(N)*sqrt(σ2_true) # μ of noise is 0

    #plt.clf()
    #plt.plot(x, t, marker="o", linestyle="None", label="training data")
    #plt.legend()
    #plt.savefig("IMG_ex_predictive_variance_ORIG.pdf")

    # Only use some set of data
    idx_use = (x .< 0.0) .| (x .> 2.0)
    x = x[idx_use]
    t = t[idx_use]

    #plt.clf()
    #plt.plot(x, t, marker="o", linestyle="None", label="training data")
    #plt.legend()
    #plt.savefig("IMG_ex_predictive_variance.pdf")

    Ndata = size(x,1)
    X = zeros(Ndata,Norder+1)
    for i in 1:Ndata
        X[i,1] = 1.0
        for n in 1:Norder
            X[i,n+1] = x[i]^n
        end
    end
    # Calculate estimate of w
    w = inv(X' * X) * X' * t # model parameter
    # Calculate estimate of σ2
    σ2 = (t'*t - t'*X*w)/Ndata

    x_test = range(-5.0, stop=5.0, step=0.1)
    NdataTest = size(x_test,1)
    X_test = zeros(NdataTest,Norder+1)
    for i in 1:NdataTest
        X_test[i,1] = 1.0
        for n in 1:Norder
            X_test[i,n+1] = x_test[i]^n
        end
    end
    μ_test = X_test*w # t_new
    σ2_test = zeros(NdataTest)
    XtXinv = inv(X'*X)
    for i in 1:NdataTest
        xnew = X_test[i,:]
        σ2_test[i] = σ2 * xnew' * XtXinv * xnew
    end

    plt.clf()
    plt.plot(x, t, marker="o", linestyle="None", label="training data", alpha=0.7)
    # We want the visualization of yerr is to be exaggerated so we used σ2_test instead
    # of its sqrt
    plt.errorbar(x_test, μ_test, yerr=σ2_test, capsize=1.0, label="test data", alpha=0.7)
    plt.legend(loc=2)
    plt.xlim(-5.5,5.5)
    plt.ylim(-1000,1000)
    plt.title("Norder = "*string(Norder))
    plt.savefig("IMG_ex_predictive_variance_Norder_"*string(Norder)*".pdf")
    plt.savefig("IMG_ex_predictive_variance_Norder_"*string(Norder)*".png", dpi=150)
end

for Norder in 1:8
    main(Norder)
end