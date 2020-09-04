#=
Simple linear regression: t = w_0 + w_1*x
Using matrix and vector notation. Using polynomial.
=#

using Printf
import DelimitedFiles: readdlm
using LinearAlgebra

import PyPlot
const plt = PyPlot
plt.rc("text", usetex=true)
plt.matplotlib.style.use("ggplot")

function eval_LH_gauss(t, x, w, σ2)
    σ = sqrt(σ2)
    μ = dot(w, x) # alternatively use w0 + w1*x
    return exp(-0.5/σ2*(t - μ)^2)/(σ*sqrt(2*pi))
end

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
    # Calculate σ2
    σ2 = (t'*t - t'*X*w)/Ndata

    return w, σ2
end


function main(Norder)
    
    data = readdlm("../../DATA/olympic100m.txt", ',')
    x = data[:,1]
    t = data[:,2]
    Ndata = size(data,1)

    # Scale x, to prevent numerical problems with large numbers
    x_min = minimum(x)
    x = x .- x_min
    x = x/4

    w, σ2 = do_fit(x, t, Norder)

    #println("w = ", w)
    logLH = 0.0
    xx = zeros(Norder+1) # input vector
    for i in 1:Ndata
        xx[1] = 1.0
        for n in 1:Norder
            xx[n+1] = x[i]^n
        end
        logLH = logLH + log(eval_LH_gauss(t[i], xx, w, σ2))
    end
    @printf("Norder = %2d logLH = %10.5f   σ2 = %8.5f\n", Norder, logLH, σ2)
end

for Norder in range(1,stop=8)
    main(Norder)
end