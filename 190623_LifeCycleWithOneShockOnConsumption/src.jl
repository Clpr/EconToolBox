"""
    src

A general module name. In this demo, it contains:
2. DiscreteMarkovChain, a type to define a discrete Markov chain
3. stationary_distribution, a function to solve stationary distribution of a `DiscreteMarkovChain`
4. approx_ar1, approximates an AR(1) process with a discrete Markov Chain
5. golden_section, Golden Section search
6. golden_section_improved, improved 3-point Golden Section search
7. DiscreteMarkovChain, discrete state Markov Chain
"""
module src
    import Distributions
    # ----------------------
    export approx_ar1
    export DiscreteMarkovChain, stationary_distribution
    export DiscreteTimeAR1
    export golden_section, golden_section_improved






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



# ----------------------- Linear interpolation & expolation
"""
    linear_function_interpolation( x0::Real, xsamples::Vector{T} where T <: Real, fsamples::Vector{T} where T <: Real )

linear interpolation on one-parameter function `f(x)` which is defined by a vector of sample `x` points
and a vector of sample `f(x)` corresponding to sample `x`;
`xsamples` should be increasingly sorted.
returns the function value evaluated at the given point `x0`.
if `x0<findmin(x)` or `x0>findmax(x)`, this function uses the linear expolation
of minimum/maximum sample `x` and the second minimum/maximum sample `x`.
"""
function linear_function_interpolation( x0::Real, xsamples::Vector{T} where T <: Real, fsamples::Vector{T} where T <: Real )
    issorted(xsamples) ? nothing : throw(ErrorException("xsamples should be pre-sorted"))
    local N = length(fsamples); local xmax = xsamples[end]; local xmin = xsamples[1]
    local x0_location = (x0 - xmin) / (xmax - xmin) * (N - 1) + 1  # locating x0
    local nearest_xsample_loc = floor(Int,x0_location)  # the nearest sample x point (on the left side) to interpolate x0
    local dist_from_nearest_xsample = x0_location - nearest_xsample_loc # the fractional distance between x0 and the nearest x point
    # NOTE: relationship: x[near] < x0 < x[near+1]

    # case: if touching the upper bound
    if (x0 > xmax) | (nearest_xsample_loc == N)  # to prevent BoundsError at index N
        res = fsamples[end]
        return res::Float64
    # case: if touching the lower bound
    elseif (x0 <= xmin) | (nearest_xsample_loc == 1)
        res = fsamples[1] - (1 - x0_location) * ( fsamples[2] - fsamples[1] )
        return res::Float64
    # case: normal case
    else
        res = (1 - dist_from_nearest_xsample) * fsamples[nearest_xsample_loc] +
            dist_from_nearest_xsample * fsamples[nearest_xsample_loc + 1]
        return res::Float64
    end # if

end # linear_function_interpolation





# ------------------------------ conventional Golden Section for 1-parameter function
"""
    golden_section(f::Function, lb::Real, ub::Real ; atol::Real = 1E-8, maxiter::Int = 5000)

conventional Golden Section search for MINIMIZATION problem.
`f` must only receive one `Real` type location parameter;
`lb` is the finite lower bound of searching; `ub` is the finite upper bound of searching;
`atol` is the absolute error to converge,
and `maxiter` sets the maximum loops to prevent dead loop.
returns a `Tuple` (in order) consisting of optimal x, optimal `f` value, a flag indicating if converged,
and in which round the algorithm ends.

testing code:
```
g(x::Real) = (x-1)^2
BenchmarkTools.@benchmark tmp = src.golden_section(g,-1.0,3.0,atol=1E-12)
```

performance tips:
```
BenchmarkTools.Trial:
  memory estimate:  96 bytes
  allocs estimate:  1
  --------------
  minimum time:     144.933 ns (0.00% GC)
  median time:      150.576 ns (0.00% GC)
  mean time:        181.232 ns (7.65% GC)
  maximum time:     71.892 μs (99.47% GC)
  --------------
  samples:          10000
  evals/sample:     852
```
"""
function golden_section(f::Function, lb::Real, ub::Real ; atol::Real = 1E-8, maxiter::Int = 5000)
    @assert(-Inf < lb < ub < Inf, "lb must be less than ub, both must be finite")
    local goldnum = 0.618033988749895  # the golden number
    local bounds = [lb,ub]  # changable bounds to update
    local left_x = bounds[1] + (1-goldnum) * (bounds[2] - bounds[1])  # left trial
    local right_x = bounds[1] + goldnum * (bounds[2]-bounds[1])  # right trial
    # initial evaluation (NOTE: using type assertion to test if `f` only returns one `Real` value)
    local left_fval::Real = f(left_x)
    local right_fval::Real = f(right_x)
    # test if both trials are defined & finite (because NaN & Inf cannot be operated rationally)
    if isnan(left_fval) | isinf(left_fval) | isnan(left_fval) | isinf(left_fval)
        throw(DomainError("NaN or Inf function value(s) found in the initial evaluation of Golden section search"))
    end # if
    # begin search
    for j in 1:maxiter
        # check convergency
        if abs(bounds[2]-bounds[1]) < atol
            local finalx::Real = (bounds[1] + bounds[2]) / 2
            return ( finalx, f(finalx), true, j )::Tuple
        end # if
        # if not converge, go on to update
        if left_fval > right_fval
            bounds[1] = left_x; bounds[2] = bounds[2]; left_x = right_x
            right_x = bounds[1] + goldnum * (bounds[2] - bounds[1])
            left_fval = f(left_x)
            right_fval = f(right_x)
        else # i.e. if left_fval <= right_fvals
            bounds[1] = bounds[1]; bounds[2] = right_x; right_x = left_x
            left_x = bounds[1] + (1-goldnum) * (bounds[2] - bounds[1])
            left_fval = f(left_x)
            right_fval = f(right_x)
        end # if
    end # for j
    # if loop normally ends, it means that the algorithm did not converge
    local finalx::Real = (left_x+right_x)/2
    return ( finalx, f(finalx), false, maxiter )::Tuple
end # golden_section




# ---------------------------- improved Golden Section search
"""
    golden_section_improved(f::Function, LLB::Float64, LB::Float64, RB::Float64 ;  TOL::Float64=1E-08, ITER=300)

improved Golden Section search, using a third point x_mid between (x_low,x_high) to improve performance.
designed for one-dim function MAXIMIZATION problem

Par:
    1. f [annonymous func]: target to search (a maximization problem), it MUST HAVE ONLY ONE PARAMETER f(x)
    1. LLB [num]: the (initial) very left bound of searching (initial)
    1. LB [num]: the (initial) left bound of searching
    1. RB [num]: the (initial) right bound of searching
    1. TOL [num]: tolerance, 1E-4 or 1E-5 is enough
    1. ITER [int]: maximum iteration times in searching

Ret:
    1. Xmin [num]: the solution

Depend:
    1. func
"""
function golden_section_improved(f::Function, LLB::Real, LB::Real, RB::Real ;  TOL::Float64=1E-08)
    # -------- DATA PROCESS ----------
    # NOTE: in Julia, the golden numebr is an integrated const (before v0.6), but we redefine a golden number for case in v1.0 and later
    GoldenNumber = 1.618033988749895
    r1 = GoldenNumber - 1; r2 = 1 - r1
    x0 = LLB; x3 = RB # initia-lize bounds
    if abs(RB-LB)<=abs(LB-LLB)
        x1 = LB; x2 = LB+r2*(RB-LB)
    else
        x2 = LB; x1 = LB-r2*(LB-LLB)
    end
    # initialization of function value (and turns a maximization to a minimization)
    f1 = - f(x1); f2 = - f(x2)
    # Searching
    while true
        if f2<f1
            x0=x1 # Update the very lower bound
            x1=x2; x2=r1*x1+r2*x3 # new left golden position
            f1=f2; f2= - f(x2)
        else
            x3=x2; x2=x1; x1=r1*x2+r2*x0; f2=f1; f1= - f(x1)
        end # if
        if abs(x3-x0) > TOL * (abs(x1)+abs(x2))
            break
        end # if
    end # while
    # Post-convergence
    Xmin = f1<=f2 ? x1 : x2

    return Xmin::Float64
end




















end # src
#
