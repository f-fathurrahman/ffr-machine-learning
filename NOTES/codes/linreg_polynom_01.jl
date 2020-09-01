#=
Simple linear regression: t = w_0 + w_1*x
Using matrix and vector notation.
=#

using Printf
import DelimitedFiles: readdlm
using LinearAlgebra

import PyPlot
const plt = PyPlot
plt.rc("text", usetex=true)
plt.matplotlib.style.use("ggplot")

# Plot per order of polynomial
function plot_data_one(x, t, w; NptsPlot=100)
    # Prepare for plot
    x_min = minimum(x)
    x_max = maximum(x)
    #
    x_plt = range(x_min, x_max, length=NptsPlot)
    #
    Norder = length(w) - 1
    t_plt = ones(NptsPlot)*w[1]
    for n in 1:Norder
        t_plt[:] = t_plt[:] + w[n+1]*x_plt.^n
    end
    plt.clf()
    plt.plot(x, t, linestyle="None", marker="o", label="data")
    plt.plot(x_plt, t_plt, label="model") # no marker
    plt.xlabel("Year")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(true)
    plt.tight_layout()
    plt.savefig("IMG_linreg_poly_"*string(Norder)*".pdf")
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
    for Norder in 1:4
        # Fit
        w = do_fit(x, t, Norder)
        #
        t_pred = do_predict(w, x)
        loss = sum( (t - t_pred).^2 )/Ndata
        @printf("Order: %3d Loss: %10.5f\n", Norder, loss)
        # Plot
        t_plt = do_predict(w, x_plt)
        plt.plot(x_plt, t_plt, label="poly-"*string(Norder))
    end
    plt.plot(x, t, linestyle="None", marker="o", label="data")
    plt.xlabel("Year (shifted and scaled)")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(true)
    plt.tight_layout()
    plt.savefig("IMG_linreg_polynom.pdf")
end

main()