"""
    UtilFuncs

This module collects many conventional utility functions
which are provided within the general form ``u(X|\\theta):\\mathbb{R}^{k}\\to\\mathbb{R}``.
where ``\\theta`` is `tuple` of parameters (in specific types of utility function, they are defined as keyword parameters), and ``k`` is the dimension of utilizable goods.

There is the contents of utility functions:
1. `u_log` - logarithm (risk netural, 1 good)
2. `u_crra` - CRRA
3. `u_cara` - CARA
4. `u_ces2` - CES (2 goods)
5. `u_cesk` - CES (k goods)
6. `u_cd1` - Cobb-Douglas (1 good)
7. `u_cd2` - Cobb-Douglas (2 goods)
8. `u_cdk` - Cobb-Douglas (k goods)
9. `u_logcd2` - Logarithm Cobb-Douglas (2 goods)
10. `u_logcdk` - Logarithm Cobb-Douglas (k goods)
11. `u_leontiefk` - Leontief (k goods)

Please refer to each specific function for detailed documentations.

This module depends on:
1. nothing

@author: Clpr @ GitHub
"""
module UtilFuncs
    export u_log, u_crra, u_cara
    export u_ces2, u_cesk
    export u_cd1, u_cd2, u_cdk
    export u_logcd2, u_logcdk
    export u_leontiefk
# =================================
"""
    u_log(x::Real)

Logarithm (risk-netural) utility function for one good
where `x` is naturally required to be positive.
The expression of logarithm utility function is:
``u(x)=\\log(x),x>0``
"""
function u_log(x::Real)
    return log(x)::Real
end # u_log


# ----------------
"""
    u_crra(x::Real ; η::Real)

CRRA (constant relative risk aversion) utility function of one good
whose expression is:
``u(x|\\eta>0) = \\frac{x^{1-\\eta}}{1-\\eta}``.
The relative risk aversion coefficient ``\\eta`` must be positive.
When ``\\eta`` is one, this function degenerates to logarithm utility function.
"""
function u_crra(x::Real ; η::Real = 1.1)
    if η<0; throw(ErrorException("requires positive eta (RRA)")); end
    η == 1 ? (return u_log(x)::Real)  :  (return (x^(1-η)/(1-η))::Real)
end # u_crra


# ----------------
"""
    u_cara(x::Real ; α::Real = 1.0)

CARA (constant absolute risk aversion) utility function of one good.
whose expression is:
``u(x|\\alpha>0) = -e^{-\\alpha x}``.
The absolute risk aversion coefficient ``\\alpha`` must be positive.
"""
function u_cara(x::Real ; α::Real = 1.0)
    if α<0; throw(ErrorException("requires positive alpha (ARA)")); end
    return (-exp(-α * x))::Real
end # u_cara


# ----------------
"""
    u_ces2(x1::Real, x2::Real ; ρ::Real = 1.0, β::Real = 1.0, α::Real = 0.5 )

CES (constant elasticity of substitution) utility function whose expression is:
``u(x_1,x_2)=(\\alpha x_1^{\\rho} + \\(1-alpha) x_2^{\\rho})^{\\beta/\\rho}``
The elasticity of substitution is ``\\frac{1}{1-\\rho}``.

Special cases:
1. when ``\\rho=1``, the function has linear/perfect substitution.
2. when ``\\rho=0``, the function becomes Cobb-Douglas function of two goods, i.e. ``u(x_1,x_2)=(x_1^\\alpha x_2^{1-\\alpha})^{\\beta}``
3. when ``\\rho=-\\infty``, the function becomes Leontief (perfect complimentary), i.e. ``u(x_1,x_2)=\\beta \\min\\{ x_1,x_2 \\}``
"""
function u_ces2(x1::Real, x2::Real ; ρ::Real = 1.0, β::Real = 1.0, α::Real = 0.5 )
    if ρ == 0
        return ( ( x1^α * x2^(1-α) )^β )::Real
    elseif ρ == -Inf
        return ( β * min(x1,x2) )::Real
    end # if
    return ( ( α*x1^ρ + (1-α)*x2^ρ )^(β/ρ) )::Real
end # u_ces2


# ----------------
"""
    u_cesk(X::Vector ; W::Vector = ones(length(X)) , β::Real = 1.0, ρ::Real = 1.0 )

CES utility function of k goods whose expression is:
``u(X)=(\\sum w_i \\log(x_i)^{\\rho})^{\\beta/\\rho}``.
requiring the length of `W` must be equal to the length of `X`.
This function does not require the summation of `W` must be 1.

Special cases:
1. when ``\\rho=1``, the function has linear/perfect substitution.
2. when ``\\rho=0``, the function becomes Cobb-Douglas function
3. when ``\\rho=-\\infty``, the function becomes Leontief (perfect complimentary)
    
"""
function u_cesk(X::Vector ; W::Vector = ones(length(X)) , β::Real = 1.0, ρ::Real = 1.0 )
    if ρ == 0
        return ( ( u_cdk(X, W=W) )^β )::Real
    elseif ρ == -Inf
        return ( β * u_leontiefk(X) )::Real
    end # if
    local U::Real = 0.0
    for (x,w) in zip(X,W)
        U += w * x ^ ρ
    end # for
    return ( U^(β/ρ) )::Real
end # u_cesk


# ----------------
"""
    u_cd1(x::Real ; θ::Real = 1.0)

Cobb-Douglas utility function of one good whose expression is:
``u(x)=x^\\theta``
"""
function u_cd1(x::Real ; θ::Real = 1.0)
    return (x^θ)::Real
end # u_cd1



# ----------------
"""
    u_cd2(x::Real ; θ::Real = 0.5)

Cobb-Douglas utility function of two good whose expression is:
``u(x_1,x_2)=x_1^\\theta\\cdot x_2^{1-\\theta}``
"""
function u_cd2(x1::Real, x2::Real ; θ::Real = 0.5)
    return (x1^θ * x2^(1-θ))::Real
end # u_cd2


# ----------------
"""
    u_cdk(X::Vector ; W::Vector = ones(length(X)))

Cobb-Douglas utility function of k goods whose expression is:
``u(X)=\\prod \\log(x_i)^{w_i}``.
requiring the length of `W` must be equal to the length of `X`.
This function does not require the summation of `W` must be 1.
"""
function u_cdk(X::Vector ; W::Vector = ones(length(X)) )
    return exp(u_logcdk(X, W = W))::Real
end # u_cdk


# ----------------
"""
    u_logcd2(x1::Real, x2::Real ; β::Real = 1.0 )

Logarithm Cobb-Douglas utility function of k goods whose expression is:
``u(x_1,x_2)=\\log(x_1)+\\beta\\log(x_2)``
"""
function u_logcd2(x1::Real, x2::Real ; β::Real = 1.0 )
    return ( log(x1) + β * log(x2) )::Real
end


# ----------------
"""
    u_logcdk(X::Vector ; W::Vector = ones(length(X)))

Logarithm Cobb-Douglas utility function whose expression is:
``u(X)=\\sum w_i\\log(x_i)``.
requiring the length of `W` must be equal to the length of `X`.
"""
function u_logcdk(X::Vector ; W::Vector = ones(length(X)) )
    local U::Real = 0.0
    for (x,w) in zip(X,W); U += w * x; end # for
    return U::Real
end # u_logcdk


# ----------------
"""
    u_leontiefk(X::Vector)

Leontief (perfect complimentary) utility function of k goods:
``u(\\mathbf{x})=\\min\\{x_1,x_2,\\dots\\}``
"""
function u_leontiefk(X::Vector)
    return findmin(X)
end # u_leontiefk








# =================================
end # UtilFuncs