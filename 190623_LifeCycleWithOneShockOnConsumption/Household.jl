"""
    Household

A module to solve a standard household life-cycle problem stated in:
[page](https://clpr.github.io/pages/blogs/190622_The_household_life_cycle_problem_with_a_continuous_medical_expenditure_shock.html)

using our custom module `src`
"""
module Household
    import src # custom module, also requires Distributions package

    export LifeCycleDatPkg



# ---------------- Types
# In this section, we define a structure to define a standard household problem.
# It helps standard I/O stream
struct LifeCycleDatPkg
    S::Int # length of life
    a1::Real # initial asset
    β::Real # utility discounting factor
    γ::Real # risk aversion of CRRA utility
    # budget: ``A_{s}a_{s+1} = B_{s}a_{s} - E_{s}c_{s} + F_{s},s=1,\\dots,S-1 ``
    A::Vector{T} where T <: Real # coef on a_{s+1}
    B::Vector{T} where T <: Real # coef on a_{s}
    𝔼E::Vector{T} where T <: Real # coef on c_{s} (expectation)
    F::Vector{T} where T <: Real # constant term
    # --------------------
    function LifeCycleDatPkg(A::Vector, B::Vector, 𝔼E::Vector, F::Vector ; a1::Real = 0, β::Real = 0.99, γ::Real = 1.5, S::Int = length(A))
        # domain check
        @assert(0<S<Inf, "S must be positive and finite integer")
        @assert(0<= a1 <Inf, "a1 must be non-negative and finite")
        @assert(0<β<Inf, "beta must be positive and finite")
        @assert(0<γ<Inf, "gamma must be positive and finite")
        @assert(all(A .> 0.0), "A_{s} must be greater than 0")
        @assert(all(B .> 0.0), "B_{s} must be greater than 0")
        @assert(all(𝔼E .> 0.0), "The expectations of E_{s} must be greater than 0")
        @assert(all(isfinite.(F)), "F_{s} must be finite")
        # length check
        @assert(S==length(A)==length(B)==length(F)==length(𝔼E), "uncompatible vector length(s)")
        return new(S,a1,β,γ, A,B,𝔼E,F)
    end # LifeCycleDatPkg
end








# ---------------- Solver
"""
    solve(m::LifeCycleDatPkg)

solves a standard household life-cycle problem, using given data package.
returns a `NamedTuple` of two elements (in order): `𝔼a`, `c`;
where `𝔼a` is the expecation path of every period's asset,
and `c` is the deterministic decision path (controller) of consumption.
"""
function solve(m::LifeCycleDatPkg)
    # case 1: S == 1 (only one period)
    # NOTE: in this case, household just spend all their asset
    #       they follow the budget: `` 0 = A * a1 - 𝔼E c1 + F ``
    if m.S == 1
        return ( 𝔼a = Float64[m.a1,], c = Float64[ (m.A[1] * m.a1 + m.F[1]) / m.𝔼E ,] )::NamedTuple
    end # if

    # case 2: S >= 2 (at least two periods)
    # NOTE: in this case, we use general solution
    # 0. define \\tilde{\\prod} B_i/A_i (len= S)
    local tmpB2A = m.B ./ m.A
    local tildeBA = Float64[]
    for x in 2:m.S
        push!(tildeBA, prod(tmpB2A[x:m.S]) )
    end # for x
    push!(tildeBA,1.0)

    # 1. define d𝔼G/dc, the derivates of compressed budgeting G on c_{s} (len= S)
    local d𝔼Gdc::Vector = tildeBA .* m.𝔼E ./ m.A
    # 2. define H̄, the Euler equation multiplier (len= S-1), H̄[s]: c[s] -> c[s+1]
    local H̄::Vector = d𝔼Gdc[1:(m.S-1)] ./ d𝔼Gdc[2:m.S]; H̄ .*= m.β; H̄ .^= -m.γ
    # 3. define M̄, the "cumulative" Euler equation multiplier (len= S), M̄[s]: c[1] -> c[s]
    local M̄::Vector = cumprod(H̄); insert!(M̄, 1, 1.0)
    println(M̄)
    # 4. define Ȳ, the denominator
    local Ȳ::Real = sum( m.𝔼E .* M̄ ./ m.A .* tildeBA )
    # 5. define X̄, the numerator
    local X::Real = m.a1 .* prod(m.B ./ m.A) .+ sum( m.F ./ m.A .* tildeBA )

    # check if X is valid
    X >= 0 ? nothing : throw(DomainError("negative consumption found"))

    # if alright, go on to get c[1], and extend the path
    local c::Vector{Float64} = (X/Ȳ) .* M̄
    # now, get the path/distribution of asset expectation in each period
    local a::Vector = Float64[m.a1,]
    for x in 1:(m.S-1); push!(a, (m.B[x]*a[x] - m.𝔼E[x]*c[x] + m.F[x])/m.A[x] ); end
    # finally, return a tuple
    return (𝔼a = a, c = c)::NamedTuple
end # solve




# ---------------- testing dataset
demodat = LifeCycleDatPkg(
    cat(fill(0.99,39),[1.0,],dims=1), # As
    fill(1.05,40), # Bs
    fill(1.19,40), # 𝔼E
    cat(fill(0.7,20),zeros(20),dims=1), # Fs
    a1 = 0.0, β = 0.99, γ = 1.5, S = 40
)










end # Household
