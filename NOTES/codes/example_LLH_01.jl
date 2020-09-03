using Printf
using LinearAlgebra

import PyPlot
const plt = PyPlot
plt.rc("text", usetex=true)
plt.matplotlib.style.use("ggplot")

function eval_LLH(t, x, w, σ2)
    σ = sqrt(σ2)
    μ = dot(w, x) # alternatively use w0 + w1*x
    return exp(-0.5/σ2*(t - μ)^2)/(σ*sqrt(2*pi))
end

function main()
    σ2 = 0.05
    w_0 = 36.416455902505334
    w_1 = -0.013330885710962845
    w = [w_0, w_1]
    x = [1.0, 1980.0]
    #
    NptsPlot = 200
    t_grid = range(9.0, stop=11.0, length=NptsPlot)
    p_grid = zeros(Float64,NptsPlot)
    for i in 1:NptsPlot
        p_grid[i] = eval_LLH(t_grid[i], x, w, σ2)
    end
    plt.clf()
    plt.plot(t_grid, p_grid)
    plt.grid(true)
    plt.xlabel("\$t\$ (seconds)")
    plt.ylabel("\$p(t|x)\$")
    plt.savefig("IMG_example_LLH.pdf")
end

main()