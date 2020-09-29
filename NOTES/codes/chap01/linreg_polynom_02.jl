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

function main(Norder)
    
    data = readdlm("../../DATA/olympic100m.txt", ',')
    x = data[:,1]
    t = data[:,2]
    Ndata = size(data,1)

    # Scale x, to prevent numerical problems with large numbers
    x_min = minimum(x)
    x = x .- x_min
    x = x/4

    NptsPlot = 100
    # Calculate again min and max of x (it is rescaled)
    x_min = minimum(x) - 5.0
    x_max = maximum(x) + 5.0
    x_plt = range(x_min, x_max, length=NptsPlot)

    plt.clf()
    w = do_fit(x, t, Norder)
    println("w = ", w)
    #
    t_pred = do_predict(w, x)
    loss = sum( (t - t_pred).^2 )/Ndata
    #
    @printf("Order: %3d Loss: %10.5f\n", Norder, loss)
    t_plt = do_predict(w, x_plt)
    # Transform the x_plt back
    x_plt2 = 4*x_plt
    x_plt2 = x_plt2 .+ minimum(data[:,1])
    plt.plot(x_plt2, t_plt, label="poly-"*string(Norder))
    #plt.plot(x_plt, t_plt, label="poly-"*string(Norder))
    plt.plot(data[:,1], t, linestyle="None", marker="o", label="data")
    plt.xlabel("Year")
    plt.ylabel("Time (s)")
    plt.ylim(9.0, 13.0)
    plt.legend()
    plt.grid(true)
    plt.tight_layout()
    plt.savefig("IMG_linreg_polynom_"*string(Norder)*".pdf")
end

for Norder in range(1,stop=8)
    main(Norder)
end