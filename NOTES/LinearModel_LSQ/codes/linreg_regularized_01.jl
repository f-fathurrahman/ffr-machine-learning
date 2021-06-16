#=
Simple linear regression: t = w_0 + w_1*x
Using matrix and vector notation.
=#

using Printf
import DelimitedFiles: readdlm
using LinearAlgebra
using Random

import PyPlot
const plt = PyPlot
plt.rc("text", usetex=true)
plt.matplotlib.style.use("ggplot")

function do_fit(x, t, Norder; λ=0.0)
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
    w = inv( X'*X + Ndata*diagm(λ*ones(Ndata)) ) * X' * t
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

Random.seed!(1234)
function main()
    x_min = 0.0
    x_max = 1.0
    x = range(x_min, x_max, step=0.2)
    # Target
    y = 2*x .- 3
    A_noise = 1.0
    t = y + A_noise*randn(size(x,1))

    plt.clf()
    NpstPlt = 100
    x_plt = range(x_min, x_max, length=NpstPlt)
    Norder = 5
    for λ in [0.0, 1e-6, 0.01, 0.1]
        w = do_fit(x, t, Norder, λ=λ)
        t_plt = do_predict(w, x_plt)
        plt.plot(x_plt, t_plt, label="\$\\lambda="*string(λ)*"\$")
    end
    plt.plot(x, t, linestyle="None", marker="o", label="data")
    plt.xlabel("\$x\$")
    plt.ylabel("\$t\$")
    plt.legend()
    plt.grid(true)
    plt.tight_layout()
    plt.savefig("IMG_linreg_regularized_01.pdf")
end

main()