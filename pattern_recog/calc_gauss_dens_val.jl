function calc_gauss_dens_val(
  μ::Vector{Float64},
  Σ::Matrix{Float64},
  x::Vector{Float64}
)
    l = size(μ)[1] # dimension
    prefactor = (2*pi)^(0.5*l) * det(Σ)^0.5
    v = (x .- μ)' * inv(Σ) * (x .- μ)

    return exp(-0.5*v)/prefactor
    #z=(1/( (2*pi)^(l/2)*det(S)^0.5) )*exp(-0.5*(x-μ)'*inv(Σ)*(x-μ));
end
