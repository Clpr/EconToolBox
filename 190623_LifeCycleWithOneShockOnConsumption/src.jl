"""
    src

A general module name. In this demo, it contains:
1. u::Function, CRRA utility with risk aversion γ
2. DiscreteMarkovChain, a type to define a discrete Markov chain
3. stationary_distribution, a function to solve stationary distribution of a `DiscreteMarkovChain`
4. approx_ar1, approximates an AR(1) process with a discrete Markov Chain
"""
module src
    import Distributions

    export u_CRRA, approx_ar1
    export DiscreteMarkovChain, stationary_distribution
    export DiscreteTimeAR1


# --------------- CRRA UTILITY
"""
    u_CRRA(c::Real ; γ::Real = 1.5, ψ::Real = 1E-12)

CRRA utility `` u(c) = [ (c+\\psi)^{1-\\gamma} - 1 ] / (1 - \\gamma) ``,
where `ψ` is a minor amount to avoid absolute zero, `c` must be non-negative and finite,
and `γ` is relative risk aversion. It must be greater than 0.
When `γ` is 1, CRRA degenerates to logarithm utility ``u(c) = \\ln(c+\\psi)``.
This function returns a `Real` value of utility.

Timing information:
```
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     1.201 ns (0.00% GC)
  median time:      1.202 ns (0.00% GC)
  mean time:        1.478 ns (0.00% GC)
  maximum time:     50.175 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1000
```
"""
function u_CRRA(c::Real ; γ::Real = 1.5, ψ::Real = 1E-12)
    # assertion
    @assert( 0 <= c < Inf, "c must be non-negative and finite" )
    @assert( 0 <= γ < Inf, "γ must be non-negative and finite" )
    @assert( ψ >= 0, "ψ must be non-negative" )
    # branch
    γ == 1 ? (return log(c + ψ)::Real) : nothing
    return ( ((c+ψ)^(1-γ) - 1) / (1-γ) )::Real
end # u_CRRA



# ------------------ TYPE: DISCRETE MARKOV CHAIN
"""
    DiscreteMarkovChain( P::Matrix{T} where T <: Real ; w::Vector = Array(1:size(T,1)) )

immutable type for discrete markov chains,
including:
1. K::Int, the number of states
2. w::Vector, state values, can be anything
2. P::Matrix{T} where T <: Real, transition matrix, square matrix, P[i,j], prob from state i to j
"""
mutable struct DiscreteMarkovChain <: Any
    K::Int  # number of states
    w::Vector # state values, can be anything
    P::Matrix{T} where T <: Real  # transition matrix
    function DiscreteMarkovChain( P::Matrix{T} where T <: Real ; w::Vector = Array(1:size(P,1)) )
        (size(P)[1] == size(P)[2])  ?  nothing  :  throw(ErrorException("non-square transition matrix received"))
        all(isapprox.( sum(P,dims = 2), 1.0))  ?  nothing  : throw(ErrorException("row-sum == 1 transition matrix received!"))
        # adjust P, making row-sum EXACT 1
        local newP = copy(P)
        for x in 1:size(P,1)
            newP[x,:] ./= sum(newP[x,:])
        end
        return new( size(P,1), w, newP )
    end
end
# -----
"""
    stationary_distribution( mc::DiscreteMarkovChain )

get the stationary distribution of a given discrete Markov Chain.
returns a `Vector{Float64}`, elements are in the order of `mc.w`.
"""
function stationary_distribution( mc::DiscreteMarkovChain )
    local tmpP = copy(mc.P)
    while true
        (sum(abs.(tmpP .- tmpP^2)) < 1E-8) ? break : nothing
        tmpP ^= 2
    end # while
    return tmpP[1,:]::Vector{Float64}
end # stationary_distribution






abstract type AbstractARProcess <: Any end
# ------------------------------- AR(1) process
"""
    DiscreteTimeAR1( ρ::Real, σ::Real ; Z̄::Real = 0.0, Z0::Union{Real,Distributions.UnivariateDistribution} = 0.0 )

An AR(1) process defined as: `` Z_{t+1} = (1-\\rho)\\cdot \\bar{Z} + \\rho Z_t + \\epsilon_t ``.
where `ρ` is the auto-reg coefficient, `Z̄` is the mean value, `σ` is the std of the error term ``\\epsilon_t``
which follows i.i.d. `N(0,σ)`, and `Z0` is the initial value/distribution of this process.
It can be a constant, or an instance of one of the subtypes of `Distributions.UnivariateDistribution`, e.g. `Normal()`.
"""
struct DiscreteTimeAR1 <: AbstractARProcess
    ρ::Real
    Z̄::Real
    Z0::Union{Real,Distributions.UnivariateDistribution}
    σ::Real
    # ---------------
    function DiscreteTimeAR1( ρ::Real, σ::Real ; Z̄::Real = 0.0, Z0::Union{Real,Distributions.UnivariateDistribution} = 0.0 )
        @assert(isfinite(ρ), "rho must be finite"); @assert(isfinite(Z̄), "the mean value must be finite")
        @assert(0<σ<Inf, "sigma must be non-negative and finite")
        isa(Z0, Real) ?  @assert(isfinite(Z0), "Z0 must be finite") : nothing
        # -----
        return new(ρ,Z̄,Z0,σ)
    end # AR1()
end # AR1










# ------------------ APPROXIMATE AR(1) WITH MARKOV CHAIN
"""
    approx_ar1(ρ::Real, σ::Real ; m::Int = 9, λ::Int = 3, Z̄::Real = 0.0 )

approximates a zero-mean AR(1) process with a discrete finite-state Markov Chain
whose states are equally divided.
The zero-mean AR(1) process is defined as:
`` Z_{t+1} = (1-\\rho)\\cdot \\bar{Z} + \\rho Z_t + \\epsilon_t ``,
where `ρ` is the auto-regression coefficient, `σ` is the standard error of ``\\epsilon_t``,
`m` is the number of the new Markov chain's states, `Z̄` is the mean of `Zt`.
`λ>0` is the absolute maximum deviation of the Markov chain's states,
it is defined as `max(z) = λσ`, where `z` is the state value of the new Markov Chain.

returns an instance of `DiscreteMarkovChain`.

This algorithm is from `(Tauchen, 1986)` and developed by `(Heer & Maussner, 2015)`.
In most cases, `m=9` and `λ=3` are enough to approximate an AR(1).
"""
function approx_ar1(ρ::Real, σ::Real ; m::Int = 9, λ::Real = 3, Z̄::Real = 0.0 )
    # assertions
    abs(ρ) < 1  ?  nothing : @warn("the AR(1) process is not stationary becuase abs(ρ) >=1")
    @assert(0 < σ < Inf, "the standard error must be greater than 0 and finite")
    @assert(1 < m, "there must be at least two states to make a Markov chain")
    @assert(0 < λ < Inf, "the absolute deviation must be greater than 0 and finite")
    @assert(isfinite(Z̄), "the mean value of Zt must be finite")
    # 1. states
    local w::Vector = zeros(m)
    w[1] = -λ * σ / √(1-ρ^2)  # minimum state value (lower bound, finite)
    local steplen::Real = -2 * w[1] / (m-1)  # step length between two states
    for x in 2:m; w[x] = w[x-1] + steplen; end;  # fill
    # 2. transition matrix
    local P::Matrix = zeros(m,m)
    local tmpNorm = Distributions.Normal() # std norm distrib
    for x in 1:m
        P[x,1] = Distributions.cdf(tmpNorm, (w[1]-ρ*w[x])/σ + 0.5 * steplen / σ )
        if m>2
            for y in 2:(m-1)
                P[x,y] = Distributions.cdf(tmpNorm, (w[y]-ρ*w[x])/σ + 0.5 * steplen / σ ) -
                    Distributions.cdf(tmpNorm, (w[y]-ρ*w[x])/σ - 0.5 * steplen / σ )
            end # for
        end # if
        P[x,m] = 1 - sum(P[x,1:(m-1)])
    end # x
    return DiscreteMarkovChain(P,w = w .+ Z̄)::DiscreteMarkovChain # add mean value to state
end # approx_ar1
# -------------------
approx_ar1(ar1::DiscreteTimeAR1 ; m::Int = 9, λ::Real = 3) = begin
    return approx_ar1(ar1.ρ, ar1.σ, m = m, λ = λ, Z̄ = ar1.Z̄)::DiscreteMarkovChain
end # approx_ar1

















end # src
#
