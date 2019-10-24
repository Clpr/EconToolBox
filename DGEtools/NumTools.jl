module src




# =================================
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
    issorted(xsamples) ? nothing : throw(ErrorException("xsamples should be pre-sorted by ascending"))
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



# =================================
"""
    utilfunc_1var(x::Real ; type::String = "logarithm", pars::Union{Tuple,NamedTuple} = ()::Tuple{} )

a dispatch of different kinds of utility functions with one argument (e.g. consumption).
supported utility functions & required parameters:
1. "logarithm" + () <- empty tuple <- ``u(x)=\\log(x)``
2. "CRRA" + (η = ,) <- relative risk aversion coefficient <- ``u(x)=\\frac{x^{1-\\eta}-1}{1-\\eta}``
3. "CARA" + (ρ = ,) <- absolute risk aversion coefficient <- ``u(x)=-e^{-\\rho x}``
4. "CD" + (θ = ,) <- share of consumption <- ``u(x)=x^{\\theta}``
"""
function utilfunc_1var(x::Real ; type::String = "logarithm", pars::Union{Tuple,NamedTuple} = ()::Tuple{} )
    if type == "logarithm"
        return u_log_1var(x)



    return nothing
end # utilfunc_1var


























end # src
#
