using LinearAlgebra
using Printf

include("calc_gauss_dens_val.jl")

function test01()
    x = [1.0, 2.0, 3.0]
    μ = [2.0, 2.0, 2.0]

    Σ = zeros(3,3)
    Σ[:,1] = [0.8, 0.0, 0.0]
    Σ[:,2] = [0.0, 0.8, 0.0]
    Σ[:,3] = [0.0, 0.0, 0.7]

    @printf("%18.10f\n", calc_gauss_dens_val( μ, Σ, x ) )
end
#test01()

function test02()
    x1 = [0.2, 1.3]
    x2 = [2.2, -1.3]
    μ = [0.0, 1.0]

    Σ = zeros(2,2)
    Σ[:,1] = [1.0, 0.0]
    Σ[:,2] = [0.0, 1.0]

    @printf("%18.10f\n", calc_gauss_dens_val( μ, Σ, x1 ) )
    @printf("%18.10f\n", calc_gauss_dens_val( μ, Σ, x2 ) )
end
#test02()


function test03()

    P1 = 0.5
    P2 = 0.5

    m1 = [1.0, 1.0]
    m2 = [3.0, 3.0]
    
    S = Matrix(1.0I, 2, 2)

    x = [1.8, 1.8]

    @printf("p1 = %18.10f\n", P1*calc_gauss_dens_val( m1, S, x ) )
    @printf("p2 = %18.10f\n", P2*calc_gauss_dens_val( m2, S, x ) )
end
test03()
