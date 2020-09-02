#=
Simple linear regression: t = w_0 + w_1*x
Using matrix and vector notation. Using polynomial.
Split data to training and validation test.
=#

using Printf
import DelimitedFiles: readdlm
using LinearAlgebra
import Statistics: mean

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

    idx_val = x .> 1979
    idx_train = x .<= 1979
    
    # Scale x, to prevent numerical problems with large numbers
    x_min = minimum(x)
    x = x .- x_min
    x = x/4

    x_val = x[idx_val]
    t_val = t[idx_val]

    NptsPlot = 100
    x_train = x[idx_train]
    t_train = t[idx_train]

    # Train
    w = do_fit(x_train, t_train, Norder)
    #println("w = ", w)
    # Predict for validation data
    t_pred = do_predict(w, x_val)
    # Evaluate loss for validation data
    #loss = mean( (t_val - t_pred).^2 )
    NdataVal = size(t_val,1)
    loss = sum( (t_val - t_pred).^2 )/NdataVal
    @printf("Order: %3d Loss: %10.5f\n", Norder, loss)
    #for i in 1:NdataVal
    #    @printf("%18.10f %18.10f\n", t_val[i], t_pred[i])
    #end
    #println()
end

for Norder in range(1,stop=8)
    main(Norder)
end