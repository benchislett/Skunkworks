# Efficient computing with the GPU in Julia 1pm-3pm Thursday July 18, 2019
# by Valentin Churavy, JuliaLab, CSAIL, MIT

using CUDAnative

function say(num)
  @cuprintf("Thread %ld says: %ld\n", threadIdx().x, num)
  return
end

# @cuda threads=4 say(42)

#=
Why should we care about GPU computing?

1. Performance

2. Powerful abstractions

=#

using CuArrays

function apply(op, a)
  i = threadIdx().x
  a[i] = op(a[i])
  return
end

x = CuArray([1., 2., 3.])

# @cuda threads=length(x) apply(y->y^2, x)

f(x) = 3x^2 + 5x + 2

# x .= f.(2 .* x.^2 .+ 6 .* x.^3 .- sqrt.(x))
# Single kernel, highly optimized

# Scalar iteration is very slow

# Not good on CuArrays
function diff_y(a, b)
  s = size(a)
  for j = 1:s[2]
    for i = 1:s[1]
      @inbounds a[i, j] = b[i, j + 1] - b[i, j]
    end
  end
end

# `CuArrays.allowscalar(false)` is useful for debugging/testing

# Always keep data on the GPU as long as possible

# Specialize with broadcasting and kernels where necessary

# ENV["JULIA_DEBUG"] = "CUDAnative"

# Benchmarking note: `CuArrays.@sync ...` to measure the code

# NVIDIA profiler:
# nvprof --profile-from-start off julia
# NVIDIA profiler graphic interface:
# nvpp julia
