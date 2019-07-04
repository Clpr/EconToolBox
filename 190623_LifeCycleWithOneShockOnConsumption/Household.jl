"""
    Household

A module to solve a standard household life-cycle problem stated in:
[page](https://clpr.github.io/pages/blogs/190622_The_household_life_cycle_problem_with_a_continuous_medical_expenditure_shock.html)

depends on our custom module `src`
"""
module Household
    import src # custom module, also requires Distributions package; used in dynamic programming

    export LifeCycleDatPkg, solve, get_a, solve_dp


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
function u_CRRA(c::Real ; γ::Real = 1.5, ψ::Real = 1E-6)
    # assertion
    @assert( 0 <= c < Inf, "c must be non-negative and finite" )
    @assert( 0 <= γ < Inf, "γ must be non-negative and finite" )
    @assert( ψ >= 0, "ψ must be non-negative" )
    # branch
    γ == 1 ? (return log(c + ψ)::Real) : nothing
    return ( ((c+ψ)^(1-γ) - 1) / (1-γ) )::Real
end # u_CRRA




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
    E::Vector{T} where T <: Real # coef on c_{s} (expectation, if there are random components)
    F::Vector{T} where T <: Real # constant term
    # --------------------
    function LifeCycleDatPkg(A::Vector, B::Vector, E::Vector, F::Vector ; a1::Real = 0, β::Real = 0.99, γ::Real = 1.5, S::Int = length(A))
        # domain check
        @assert(0<S<Inf, "S must be positive and finite integer")
        @assert(0<= a1 <Inf, "a1 must be non-negative and finite")
        @assert(0<β<Inf, "beta must be positive and finite")
        @assert(0<γ<Inf, "gamma must be positive and finite")
        @assert(all(A .> 0.0), "A_{s} must be greater than 0")
        @assert(all(B .> 0.0), "B_{s} must be greater than 0")
        @assert(all(E .> 0.0), "The expectations of E_{s} must be greater than 0")
        @assert(all(isfinite.(F)), "F_{s} must be finite")
        # length check
        @assert(S==length(A)==length(B)==length(F)==length(E), "uncompatible vector length(s)")
        return new(S,a1,β,γ, A,B,E,F)
    end # LifeCycleDatPkg
end

# ---------------- testing dataset
demodat = LifeCycleDatPkg(
    cat(fill(0.99,39),[1.0,],dims=1), # As
    fill(1.05,40), # Bs
    fill(1.19,40), # E
    cat(fill(0.7,20),zeros(20),dims=1), # Fs
    a1 = 0.0, β = 0.99, γ = 1.5, S = 40
)





# ------------------------------------------------------------------------------
# NOTE: the following section is for ANALYTICAL solution WITHOUT borrowing constraints
# ---------------- Analytical Solver (no borrowing constraint)
"""
    solve(m::LifeCycleDatPkg)

solves a standard deterministic household life-cycle problem, using given data package.
returns a `NamedTuple` of two elements (in order): `a`, `c`;
where `a` is the expecation path of every period's asset,
and `c` is the deterministic decision path (controller) of consumption.
"""
function solve(m::LifeCycleDatPkg)
    # case 1: S == 1 (only one period)
    # NOTE: in this case, household just spend all their asset
    #       they follow the budget: `` 0 = A * a1 - E c1 + F ``
    if m.S == 1
        return ( a = Float64[m.a1,], c = Float64[ (m.A[1] * m.a1 + m.F[1]) / m.E ,] )::NamedTuple
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

    # 1. define d𝔼G/dc, the derivatives of compressed budgeting G on c_{s} (len= S)
    local d𝔼Gdc::Vector = tildeBA .* m.E ./ m.A
    # 2. define H̄, the Euler equation multiplier (len= S-1), H̄[s]: c[s] -> c[s+1]
    local H̄::Vector = d𝔼Gdc[1:(m.S-1)] ./ d𝔼Gdc[2:m.S]; H̄ .*= m.β; H̄ .^= -m.γ
    # 3. define M̄, the "cumulative" Euler equation multiplier (len= S), M̄[s]: c[1] -> c[s]
    local M̄::Vector = cumprod(H̄); insert!(M̄, 1, 1.0)
    println(M̄)
    # 4. define Ȳ, the denominator
    local Ȳ::Real = sum( m.E .* M̄ ./ m.A .* tildeBA )
    # 5. define X̄, the numerator
    local X::Real = m.a1 .* prod(m.B ./ m.A) .+ sum( m.F ./ m.A .* tildeBA )

    # check if X is valid
    X >= 0 ? nothing : throw(DomainError("negative consumption found"))

    # if alright, go on to get c[1], and extend the path
    local c::Vector{Float64} = (X/Ȳ) .* M̄
    # now, get the path/distribution of asset expectation in each period
    local a::Vector = Float64[m.a1,]
    for x in 1:(m.S-1); push!(a, (m.B[x]*a[x] - m.E[x]*c[x] + m.F[x])/m.A[x] ); end
    # finally, return a tuple
    return (a = a, c = c)::NamedTuple
end # solve




# ----------------- compute a[s], s=2 to S+1 when given c[s], s=1 to S
"""
    get_a(d::LifeCycleDatPkg, C::Vector)

when `s` is 1, `d.a1` will be directly returned;
otherwise, `length(C) >= s-1` requried
"""
function get_a(d::LifeCycleDatPkg, s::Int, C::Vector)
    if s==1; return (d.a1)::Real; end
    # if not a[1]
    @assert(length(C) >= (s-1), "length(C) >= s-1 requried")
    # computing
    local as::Real = d.a1
    for x in 1:(s-1)
        as = ( d.B[x] * as - d.E[x] * C[x] + d.F[x] ) / d.A[x]
    end # for x
    return as::Real
end # get_a








# =============== STD DP SOLVER (DETERMINISTIC)
# NOTE: using the same datastructure LifeCycleDatPkg

# --------------- small type to pass parameters
"""
    BellmanParSet

a simple structure to pass in the parameters required by `bellman_val()`
"""
struct BellmanParSet <: Any
    Athis::Real; Bthis::Real; Ethis::Real; Fthis::Real # pars in budget, used to compute c[s]
    β::Real # utility discounting factor
    γ::Real # risk aversion
    # ------- define a constructor to allow any-order parameter list
    function BellmanParSet(; Athis::Real = 0.0, Bthis::Real = 0.0, Ethis::Real = 0.0, Fthis::Real = 0.0, β::Real = 1.0, γ::Real = 2.0)
        return new(Athis,Bthis,Ethis,Fthis,β,γ)
    end
end # BellmanParSet
# --------------- a convenient reloaded external constructor
BellmanParSet(m::LifeCycleDatPkg, s::Int) = begin
    return BellmanParSet(Athis = m.A[s], Bthis = m.B[s], Ethis = m.E[s], Fthis = m.F[s], β = m.β, γ = m.γ)
end


# -------------- minor function: c[s] = [ B[s]a[s] + F[s] - A[s]a[s+1] ] / E[s]
"""
    get_cthis( athis::Real, anext::Real, pars::BellmanParSet )

computes c[s] when given a[s], a[s+1] and other parameters in the inter-temporal budget constraint.
returns a `Float64`
"""
get_cthis( athis::Real, anext::Real, pars::BellmanParSet ) = begin
    return ( (pars.Bthis * athis + pars.Fthis - pars.Athis * anext) / pars.Ethis )::Float64
end # get_cthis
# -------------- function of the value of value functions
"""
    bellman_val( athis::Real, anext::Real, anext_vec::Vector, vnext_vec::Vector, pars::BellmanParSet )

computes `u(c[s]) + βV_{s+1}(a[s+1])` (no `max` yet!) when given a[s] (`athis`),
a[s+1] (`anext`), the INCREASINGLY-SORTED discrete function series ``(a_{s+1},V_{s+1})`` (`anext_vec` & `vnext_vec`),
and other parameters (passed in `pars`) used in inter-temporal budgeting.
returns a `Float64`. If computed `c[s]` is infeasible (<0), this function returns `-9.99E99`

Please note that, `next_vec` & `vnext_vec` actually together define the function ``V_{s+1}(a_{s+1})``,
as independent variable and function value. They are in the form of discrete 2-dim points.
"""
function bellman_val( athis::Real, anext::Real, anext_vec::Vector, vnext_vec::Vector, pars::BellmanParSet )
    # get consumption c[s]
    local cthis::Real = get_cthis(athis,anext,pars)
    # c[s] is valid? (>=0 required by definition)
    if cthis < 0.0
        return (-9.99E99)::Float64
    else
        return ( u_CRRA(cthis, γ = pars.γ) + pars.β * src.linear_function_interpolation(anext, anext_vec, vnext_vec) )::Float64
    end # if
end # bellman_val


# ------------------------- standard solver
"""
    solve_dp(m::LifeCycleDatPkg ; arange::Tuple{Float64,Float64} = (0.0,sum(m.F) + a.a1), gridnum::Int = 200 )

standard solver for a deterministic Bellman equation which uses CRRA utility (with parameter γ),
where households do not intend to leave bequests.
The bellman equation follows the following inter-temporal budget:
``
A_s k_{s+1} = B_s k_{s} - E_{s} c_{s} + F_{s}, s=1,\\dots,S
``
where A,B,E,F are given parameters (deterministic);
for E, it is defined as member `E` in `LifeCycleDatPkg`.

Note: we use `sum(m.F)+a.a1` as the guess of ``a_s`` upper bound because someone
cannot consume more than his/her life-time total income when meeting borrowing constraint (`arange[1]=0.0`).
"""
function solve_dp(m::LifeCycleDatPkg ; arange::Tuple{Float64,Float64} = (0.0,sum(m.F) + m.a1), gridnum::Int = 200 )
    # validation
    @assert(-Inf < arange[1] < arange[2] < Inf, "arange must be finite and the lower bound must be less than the upper bound")
    @assert(1 < gridnum < Inf, "the number of grid points must be positive and finite integer")
    # case: m.S == 1 (only one period, no need to compute)
    if m.S == 1
        tmppars = BellmanParSet(m,1)
        return ( a = Float64[m.a1,], c = Float64[ get_cthis(m.a1,0.0,tmppars) ,] )::NamedTuple
    end # if
    # if not, go on
    # malloc
    local agrid = Array(LinRange(arange[1],arange[2],gridnum)) # griding vector of a_{s}
    local optv = zeros(gridnum,m.S) # policy function(s) component matrix, for value functions' values
    local opta = zeros(gridnum,m.S) # policy function(s) component matrix, for optimal asset levels
    local optc = zeros(gridnum,m.S) # policy function(s) component matrix, for optimal consumption
    local apath = zeros(m.S) # solved path for asset, the 1st element is m.a1
    local cpath = zeros(m.S) # solved path for consumption
    # ---------- let`s go
    # period: s = m.S
    idxS = m.S
        for x in 1:gridnum
            # first, fill the last column (policy function of period S) with griding asset(s)
            opta[x, m.S] = agrid[x]
            # then, compute consumption for every possible a_{S}; NOTE: no intend to leave bequest
            tmppars = BellmanParSet(m, idxS)
            optc[x, m.S] = get_cthis(agrid[x], 0.0, tmppars)
            # finally, use u(c[S]) as V_{S}
            optv[x, m.S] = u_CRRA(optc[x,m.S], γ = tmppars.γ)
        end # for x
    # period: 1 <= s < m.S (NOTE: this block compatible with the case that m.S == 2)
    for idxS in (m.S-1):-1:1  # NOTE: on this layer, we compute policy functions for every period
        for x in 1:gridnum  # NOTE: on this layer, we compute policy function points for every possible a[s]
            # first, define the parameter set used in this loop of x
            tmppars = BellmanParSet(m, idxS)
            # then, define a one-parameter function for Bisection search later
            # NOTE: we have to define it here because it depends on `x` and `idxS`
            local afunc(anext::Real) = bellman_val(agrid[x],  anext,  agrid,optv[:,idxS+1],tmppars)
            # next, compute a temporary value function series for griding a[s]
            # NOTE: this temp series is exactly the values of ``[u(c) + βV_{s+1}(a_{s+1})]`` for every possible a_{s+1};
            #       we will find its maxmimum as the value of ``V_{s}`` later, i.e. the Bellman equation's definition
            local tmpv = Float64[]
            for j in 1:gridnum; push!( tmpv, afunc(agrid[j]) ); end # for j
            # now, let's find the maximum of `tmpv`
            local tmploc = findmax(tmpv)[2]  # locate the maximum
            if (tmploc == 1) | (tmploc == gridnum)
                isfinite(tmpv[tmploc]) ? nothing : throw(DomainError("infeasible DP problem found when evaluating Bellman equation in period $(idxS)"))
                opta[x,idxS] = agrid[tmploc]
            else
                # if not the 1st or the last element of griding asset series, we use Golden Section search to get a more precise a[s] in a possible range
                # NOTE: assuming we find the maximum at location j, then we search in range (agrid[j-1], agrid[j+1])
                # NOTE: the golden_section() function checks if gold_lb or gold_ub is infinite or NaN; if infinite or NaN, an error will be thrown by the function
                local gold_lb = agrid[tmploc-1]; local gold_ub = agrid[tmploc+1]
                local tmpres = src.golden_section(afunc, gold_lb, gold_ub, atol = 1E-8, maxiter = 5000)
                # check if the solution is feasible
                if isfinite(tmpres[1]) & isfinite(tmpres[2]) # check optimal a[s+1] and V[s]
                    opta[x,idxS] = tmpres[1] # optimal a[s+1] (please note where I save it in `opta`)
                else
                    throw(ErrorException("no solution, stopped at evaluating the policy function of period $(idxS) at asset level $(agrid[x]); the stopping optimal V is $(tmpres[2])"))
                end # if
            end # if tmploc

            # finally, we evalute the (interpolated/precise/accurate/continuous/real) ``V_{s}`` and ``c^*_{s}`` and save
            # NOTE: we need this step because we (might) have get a non-grid asset point (opta[x,idxS]) as our optimum
            optv[x,idxS] = afunc(opta[x,idxS])
            optc[x,idxS] = get_cthis(agrid[x], opta[x,idxS], tmppars)

        end # for x
    end # for idxS

    # now, let's extract optimal paths conditional on a[1] (as input parameter in `m`)
    apath[1] = m.a1
    for x in 1:(m.S-1)
        # query & interpolate a[s+1] with src.linear_function_interpolation
        # NOTE: dont forget the policy function is defined as such a mapping: agrid (a[s]) --> opta[:,x] (optimal a[s+1])
        #       this helps to better understand `linear_function_interpolation()`
        apath[x+1] = src.linear_function_interpolation(apath[x], agrid, opta[:,x])
        # using inter-temporal budget to get c[s+1]
        tmppars = BellmanParSet(m, x)
        cpath[x] = get_cthis(apath[x],apath[x+1],tmppars)
    end # for x
    # finally, get c[S] (because we have the boundary condition that a[S+1]=0, the computing of c[S] needs to be separated from the above loop)
    tmppars = BellmanParSet(m, m.S)
    cpath[m.S] = get_cthis(apath[m.S],0.0,tmppars)


    # finally, returns the results in a NamedTuple
    return ( a = apath, c = cpath )::NamedTuple
end # solve_dp














end # Household
