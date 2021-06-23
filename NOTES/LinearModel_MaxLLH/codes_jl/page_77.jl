using Printf
import Statistics: mean

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

function main()
    w_true = [-2.0, 3.0]
    σ2_true = 0.5^2
    Nsample = 20
    x = rand(Nsample)  # sample from uniform distribution
    # Generate
    t = zeros(Nsample)
    for i in 1:Nsample
        t[i] = w_true[1] + w_true[2]*x[i] + σ2_true*randn()
    end

    # fit
    w = do_fit(x, t, 1)
    x_plt = range(minimum(x), stop=maximum(x), length=100)
    y_plt = do_predict(w, x_plt)

    plt.clf()
    plt.plot(x, t, label="data", linestyle="None", marker="o")
    plt.plot(x_plt, y_plt, label="model")
    # plt.scatter also can be used
    plt.savefig("IMG_page77.pdf")
end

main()