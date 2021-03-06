#=
Simple linear regression: t = w_0 + w_1*x
Using matrix and vector notation.
=#

import DelimitedFiles: readdlm
using LinearAlgebra

function main()
    data = readdlm("../../DATA/olympic100m.txt", ',')
    x = data[:,1]
    t = data[:,2]
    Ndata = size(data,1)
    # Build X matrix
    X = zeros(Ndata,2)
    for i in 1:Ndata
        X[i,1] = 1.0
        X[i,2] = x[i]
    end
    # Calculate w
    w = inv(X' * X) * X' * t
    println("w = ", w)
end

main()