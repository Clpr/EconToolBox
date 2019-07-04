"""
    Household_StochasticDP

Stochastic DP for a standard household life-cycle model with one shock on consumption.
developed based on `Household.jl` and `src.jl`
"""
module Household_StochasticDP
    import src # math dependency
    import Household # for basic methods


# --------------------- CRRA UTILITY
function u_CRRA(c::Real ; γ::Real = 1.5, ψ::Real = 1E-6)
    # assertion
    @assert( 0 <= c < Inf, "c must be non-negative and finite" )
    @assert( 0 <= γ < Inf, "γ must be non-negative and finite" )
    @assert( ψ >= 0, "ψ must be non-negative" )
    # branch
    γ == 1 ? (return log(c + ψ)::Real) : nothing
    return ( ((c+ψ)^(1-γ) - 1) / (1-γ) )::Real
end # u_CRRA
































end # Household_StochasticDP
