"""
    TypeCore

Defines basic Types and DataStructures.
Methods binded with Types are separately defined in other modules.
It includes:
1. Stochastic Tools
    1. AbstractMarkovChain <: Any
    2. OrdinaryMarkovChain <: AbstractMarkovChain
2. 
"""
module TypeCore
    import LinearAlgebra

    export OrdinaryMarkovChain, AR1
# ==============================




# ------------------
"""
    AbstractMarkovChain

The abstract type of Markov Chains.
"""
abstract type AbstractMarkovChain <: Any end
# ------------------
"""
    OrdinaryMarkovChain(P::Matrix{Float64} ; S::Vector{T} where T <: Real = 1:size(P)[1])

Discrete time, finite number of states,
states in number, constant transition matrix,
Markov Chain.
Receives: `P` - one-step transition matrix `P[i,j]` (prob from state i to state j);
`S` - vector of state values, using 1,2,... in default.

Requires `P`, the transition probability matrix must be square.
An `OrdinaryMarkovChain` must have a stationary distribution.
"""
struct OrdinaryMarkovChain <: AbstractMarkovChain
    N::Int  # the number of states
    S::Vector{T} where T <: Real  # states (in numbers)
    P::Matrix{Float64}  # (one-step) transition matrix P_{ij}, prob from state i to state j
    function OrdinaryMarkovChain(P::Matrix{Float64} ; S::Vector{T} where T <: Real = [x for x in 1:size(P)[1]])
        local Pdim::Tuple = size(P)
        local Pnew::Matrix = zeros(Float64, Pdim)
        @assert(Pdim[1] == Pdim[2], "requires a square matrix")
        @assert(Pdim[1] == length(S), "dimension mismatch between state vector and transition matrix")
        for z in 1:Pdim[1] # rescale each row to make row-sum is exact one
            Pnew[z,:] = LinearAlgebra.normalize(P[z,:], 1) # the 2nd par indicates linearly normalization
        end # for
        return new(Pdim[1], S, Pnew)
    end # constructor
end # OrdinaryMarkovChain


# ---------------------
"""
    AR1

AR(1) process ``x_{t+1}=c+\\rho x_{t}+\\varepsilon_{t}``
where ``\\varepsilon_{t}\\sim N(0,\\sigma)``.
"""
struct AR1 <: Any
    c::Real
    rho::Real
    sigma::Real
end # AR1




















# ==============================
end # TypeCore