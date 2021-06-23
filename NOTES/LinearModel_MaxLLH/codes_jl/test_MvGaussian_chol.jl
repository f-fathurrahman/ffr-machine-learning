using Random
Random.seed!(1234)

# Not yet finished
function gaussian_pdf(μ::Vector{Float64}, Σ::Matrix{Float64}, N::Int64)
    L = cholesky(Σ)
    q = randn(length(μ))
    display(L); println()
end

function main()
    Σ = [1.0 2.0; 2.0 4.0]
    println("Σ = ", Σ)
    μ = [1.0, 3.0]
    N = 3
    gaussian_pdf(μ, Σ, N)
end

main()