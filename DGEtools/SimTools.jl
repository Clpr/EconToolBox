"""
    SimTools

Simulation toolbox used in DGE models.
The table of content is:
1. Private Types/Data Structures
2. Samplings
    1. `lhsnorm()` - Ordinary Latin Hypercube Sampling on normal distribution
    2. `lhsmnorm()` - Ordinary Latin Hypercube Sampling on multinvariate normal distribution
    3. `lhsempirical()` - Ordinary Latin Hypercube Sampling on an one-way empirical distribution
3. Algorithms
    1. `cdf_empirical()` - build empirical distribution (cdf) when given samples from a univariate continuous distribution
4. Sample/Testing Data



This module depends on:
1. `Distributions.jl`
2. `Random.jl` (std lib)
3. `LinearAlgebra.jl` (std lib)
4. `StatsBase.jl`
5. `StatsFuns.jl`
6. `Dierckx.jl` - wraper of classical Dierckx interpolation library in Fortran

@author: Clpr @ GitHub
"""
module SimTools
    import StatsFuns, Distributions, StatsBase, Dierckx
    import Random, LinearAlgebra  # std lib

    export lhsnorm, lhsmnorm, lhsempirical
    export cdf_empirical
# =============================

# ------------------
"""
    lhsnorm(N::Int ; μ::Real = 0.0, σ::Real = 1.0)

Ordinary Latin Hypercube Sampling (LHS) on Normal distribution.
Returns a `Vector{Float64}` with `N` elements
"""
function lhsnorm(N::Int ; μ::Real = 0.0, σ::Real = 1.0)
    @assert((N>0)&(σ>0), "invalid N or std")
    local Res::Vector{Float64} = rand(N)
    for x in 1:N
        Res[x] = StatsFuns.norminvcdf( μ, σ, (Res[x] + x - 1)/N )
    end # for
    Random.shuffle!(Res)  # shuffle the sample
    return Res::Vector{Float64}
end # lhsnorm


# ------------------
"""
    lhsmnorm(N::Int ; μ::Vector = [1.0, 1.0], Σ::Matrix = [1.0 0; 0 1.0] )

A wraper of `lhsmnorm(::Int, Distributions.AbstractMvNormal)`.
Receives custom mean vector `μ` and covariance matrix `Σ`.
"""
function lhsmnorm(N::Int ; μ::Vector = [1.0, 1.0], Σ::Matrix = [1.0 0; 0 1.0] )
    return (lhsmnorm(N, Distributions.MvNormal(μ,Σ)))::Matrix{Float64}
end # lhsmnorm
# -------------------
"""
    lhsmnorm(N::Int, D::Distributions.AbstractMvNormal)

Ordinary Latin Hypercube Sampling (LHS) on multinvariate normal distribution.
Returns a `Matrix{Float64}` with `N` rows/samples in which each row of the matrix is a sample.
The dimension of samples depends on the input mean value vector `μ` and covariance matrix `Σ`.

This algorithm follows:
1. Iman, R. L., and W. J. Conover. 1982. A Distribution-free Approach to Inducing Rank Correlation Among Input Variables. Communications in Statistics B 11:311-334
2. Zhang, Y. , & Pinder, G. . (2003). Latin hypercube lattice sample selection strategy for correlated random hydraulic conductivity fields. Water Resources Research, 39(8).
3. My blog: https://clpr.github.io/pages/blogs/190420_LHSonMvNormal.html
"""
function lhsmnorm(N::Int, D::Distributions.AbstractMvNormal)
    # init
    local Ddim::Int = Distributions.dim(D.Σ)
    local Res::Matrix{Float64} = rand(Float64, N, Ddim)
    local Rnk1::Matrix{Int} = zeros(Int, N, Ddim) # (row) ranks of each column for Res with correlation
    local EachDimStd = sqrt.(LinearAlgebra.diag(D.Σ)) # std of each marginal distribution
    # loop to re-scale & inverse to a MvNormal without correlation (each dim is independent)
    # NOTE: just use our lhsnorm() for univariate normal distribution which auto shuffles the results
    for z in 1:Ddim
        Res[:,z] = lhsnorm(N, μ=D.μ[z], σ=EachDimStd[z])
    end # for
    # compute R (Ddim * Ddim size), the correlation matrix (not cov) of the Res without correlation
    # NOTE: when N*P is large, it is a good idea to use a eye(Ddim) to approximate R
    local Rmat::Matrix{Float64} = (N > 1000) ? LinearAlgebra.diagm(0 => ones(Ddim)) : StatsBase.cor(Res)
    # get a new sample matrix Res1
    # NOTE: we do not specially declare Q,P but integrate them to the computing of X1
    # the complier is smart enough to optimize it for least memeory & time costs
    local Res1 = Res * transpose( LinearAlgebra.cholesky( Distributions.cor( D ) ).L * inv(LinearAlgebra.cholesky( Rmat ).L) )
    # record ranks of each column of Res1, then rearrange Res's columns (one by one) according to Res1's col-ranks
    for z in 1:Ddim
        # first, record X1's rank values
        Rnk1[:,z] = StatsBase.ordinalrank(Res1[:,z])
        # then, sort X0's column, the index is the rank value
        Res[:,z] = sort( Res[:,z] )
        # finally, rearrange indices according to Rnk1
        Res[:,z] = Res[Rnk1[:,z], z]
    end # for
    # return (we have already shuffled columns)
    return Res::Matrix{Float64}
end # lhsmnorm
 

# --------------------
"""
    lhsempirical(N::Int, X::Vector)

Uses Ordinary Latin Hypercube Sampling to sample on a continuous empirical distribution when given a sample vector `X`.
`N` is the number of samples to generate.
Using one-way (cubic) spline interpolation to interpolate the constructed empirical cdf (from sample vector `X`).
Returns a `Vector{Float64}`.

Note: usually, this function is more efficient when using relatively small size of samples to generate large size of samples.
"""
function lhsempirical(N::Int, X::Vector)
    # NOTE: I write the whole proc to estimate the empirical cdf to avoid unnecessary dependency
    # first, rebuild the distribution (empirical cdf)
    local Demp::Tuple = cdf_empirical(X) # this function is defined below
    # then, sampling on Uniform(0,1)
    local Res::Vector{Float64} = rand(N)
    # then, construct a spline interpolation instance
    # NOTE: we use Dierckx.jl, a wraper of Dierckx of Fortran; it is high-performance
    # NOTE: because we will use the inverse cdf, then we use cdf as X but realized values as Y here
    local CdfEmp::Dierckx.Spline1D = Dierckx.Spline1D(Demp[2], Demp[1])
    # then, do ordinary LHS (inverse cdf)
    for x in 1:N
        # rescale Uniform(0,1) to Uniform((x-1)/N, x/N)
        Res[x] = (Res[x] + x - 1)/N  
    end # for
    # NOTE: "=" costs similar memory as ".=" but faster
    Res = CdfEmp.(Res)
    Random.shuffle!(Res)  # shuffle the sample
    return Res::Vector{Float64}
end # lhsempirical




















# ---------------------
"""
    cdf_empirical(X::Vector)

Rebuilds an empirical cdf when given a sample vector `X`
from a univariate distribution.
Returns a tuple (image) `(X,F)` in which
the 1st member is the input `X` (ascendingly sorted) vector, and `F` is cdf values vector.
"""
function cdf_empirical(X::Vector)
    local Xdim::Int = length(X)
    @assert(Xdim>1, "too few samples (only one!)")
    local Xsort::Vector = sort(X)
    local CdfEmp = Float64[0.0]
    # then, build empirical cdf by estimating empirical quantiles
    # considering that in practice, there will not be so many replicated samples from a continuous distribution,
    # then, using a nested forward-looking searching at every `z` (as follows) is still much more economic than applying conditional filters on `X`
    for z in 2:Xdim
        # this handles the end of loop and the case that Xdim==2
        if z == Xdim
            push!(CdfEmp, 1.0)
            break # out of loop
        end # if
        # else, go on:
        if Xsort[z] != Xsort[z+1] # if there is no same-value in the next ascendingly-ordered sample value
            push!(CdfEmp, z/Xdim) # using formula F(x)= (number of samples which are less than or equal to current x)/N
        else # in this case, we need to consider same-value cases
            local m::Int = 1
            while z+m <= Xdim
                # if no more same-value sample, end loop
                (Xsort[z] == Xsort[z+m]) ? (m += 1) : break
            end # while
            # because we added an extra one in the above while loop, then we need to minus it now
            push!(CdfEmp, (z+m-1)/Xdim)
        end # if
    end # for
    return (Xsort,CdfEmp)::Tuple{Vector{Float64},Vector{Float64}}
end # cdf_empirical





















# =============================
end # SimTools