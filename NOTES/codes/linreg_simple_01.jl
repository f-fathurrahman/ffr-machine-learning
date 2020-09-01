#=
Simple linear regression: t = w_0 + w_1*x
=#

import DelimitedFiles: readdlm
import Statistics: mean

import PyPlot
const plt = PyPlot
plt.rc("text", usetex=true)
plt.matplotlib.style.use("ggplot")

function main()
    data = readdlm("../../DATA/olympic100m.txt", ',')
    x = data[:,1]
    t = data[:,2]

    xbar = mean(x)
    tbar = mean(t)

    w_1 = (mean(x.*t) - xbar*tbar)/(mean(x.^2) - xbar^2)
    w_0 = tbar- w_1*xbar
    println("w_0 = ", w_0)
    println("w_1 = ", w_1)

    # Prepare for plot
    x_min = minimum(x)
    x_max = maximum(x)
    x_plt = range(x_min, x_max, length=100)
    t_plt = w_0 .+ w_1*x_plt

    plt.clf()
    plt.plot(x, t, linestyle="None", marker="o", label="data")
    plt.plot(x_plt, t_plt, label="model") # no marker
    plt.xlabel("Year")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(true)
    plt.tight_layout()
    plt.savefig("IMG_linreg_simple.pdf")
end

main()