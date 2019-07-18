# Open Source Contribution in Julia (OIST) 10am-12pm Thursday July 18, 2019
# by Valentin Churavy, JuliaLab, CSAIL, MIT

#=
Open Source
Benefits:
- Helps other people
- Free
- No privatization

Downsides:
- Can be a large time sink
- Maintaining can be difficult
- Requires a thick skin

The Structure of a Contribution
https://opensource.guide/how-to-contribute/

1. Formulate a need: Why are you doing this? What do you stand to gain?
2. Communicate: Open an issue, write on the mailing list
3. Write a pull-request that is up to the standards of the project
4. Be kind to maintainers and have patience
=#

#=
How to contribute to a Julia package

1. `]dev PackageName`
2. `cd ~/.julia/dev/PackageName`
3. `git checkout -b bugfix`
4. Make changes
5. Test changes `]test PackageName`
6. `git commit`
=#

#=
How to contribute to Julia
Build from source

1. `git clone https://github.com/JuliaLang/julia`
2. `cd julia && make -j`
3. Make changes
4. `make testall` or `make -C test threads`
5. As above
=#

# https://pretalx.com/juliacon2019/talk/

# Unitful.jl
# Measurements.jl

# StaticArrays.jl
# StructArrays.jl
# Images.jl and Colors.jl

function mySqrt(x; N = 10)
  t = (1+x)/2
  for i=2:N
    t = (t + x/t)/2
  end
  return t
end
α = π
# @assert mySqrt(α) ≈ √α

# Define a derivative

struct D <: Number
  f::Tuple{Float64, Float64}
end

import Base: +, /, convert, promote_rule
+(x::D, y::D) = D(x.f .+ y.f) # d(f+g)/dx = df/dx + dg/dx
/(x::D, y::D) = D((x.f[1]/y.f[1], (y.f[1]*x.f[2] - x.f[1] * y.f[2]) / y.f[1]^2))
convert(::Type{D}, x::Real) = D((x,zero(x)))
promote_rule(::Type{D}, ::Type{<:Number}) = D

# mySqrt(D((α,1))), (√α, 0.5/√α)

using UnicodePlots
using ForwardDiff

f(x) = 1 / (1 + exp(-x))
g(x) = ForwardDiff.derivative(f, x)
h(x) = ForwardDiff.derivative(g, x)

lineplot([f, g], width=100, height=40)
