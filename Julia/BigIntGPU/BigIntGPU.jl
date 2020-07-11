
module BigIntGPU
  using CuArrays, CUDAnative
  import Base:length, zero, +

  export IntGPU, uadd, toBig, fromBig
  
  const LIMB_TYPE = UInt64
  const LIMB_MAX = typemax(LIMB_TYPE) ÷ 2
  const NUM_THREADS = 128

  mutable struct IntGPU
    sign::Bool
    d::CuArray{LIMB_TYPE,1}
  end

  function IntGPU(x::LIMB_TYPE)
    @assert x < LIMB_MAX
    return IntGPU(x >= 0, CuArray([LIMB_TYPE(x)]))
  end

  function idx()
    return (blockIdx().x - 1) * blockDim().x + threadIdx().x
  end

  function getBlocks(n::Int)
    Int(ceil(n / NUM_THREADS))
  end

  function prefix_sum_kernel!(x, op::Function=+)
    i = idx() 
    n = UInt32(length(x)/2)
    if (i > n)
      return nothing
    end

    pin, pout = 1, 0
    offset = 1
    while offset < n
      pout = 1 - pout
      pin = 1 - pin
      if i > offset
        @inbounds x[pout * n + i] = op(x[pin * n + i], x[pin * n + i - offset])
      else
        @inbounds x[pout * n + i] = x[pin * n + i]
      end
      sync_threads()
      offset *= UInt32(2)
    end
    @inbounds x[pin * n + i] = x[pout * n + i]

    return nothing
  end
  
  function oplus(new, old)
    if (new == 0)
      return 0
    elseif (new == 1)
      return old
    else
      return 2
    end
  end
  
  CuZeros(n) = CuArrays.zeros(LIMB_TYPE, n)
  CuEmpty(n) = CuArray{LIMB_TYPE}(undef, n)
  CuEmpty(T, n) = CuArray{T}(undef, n) # where T

  length(x::IntGPU) = Base.length(x.d)
  zero(x::IntGPU) = Base.zero(x, length(x))
  zero(x::IntGPU, y::Int) = IntGPU(true, CuZeros(y))
 
  function resize_kernel(x::CuArray, y::CuArray)
    i = idx()
    if (i > length(x))
      if (i > length(y))
        return nothing
      end
      @inbounds y[i] = 0
    else
      @inbounds y[i] = x[i]
    end
  end

  function resize(x::CuArray, n::Int)
    out = CuEmpty(n)
    @cuda blocks=getBlocks(n) threads=NUM_THREADS resize_kernel(x, out)
    return out
  end

  function uadd_sum_kernel(a_d::T, b_d::T, out_d, carry) where T
    i = idx()
    if (i * 2 > length(carry))
      return nothing
    end

    a_i::eltype(a_d) = 0
    b_i::eltype(b_d) = 0
    if (i <= length(a_d))
      @inbounds a_i = a_d[i]
    end

    if (i <= length(b_d))
      @inbounds b_i = b_d[i]
    end

    res = a_i + b_i
    if (res >= LIMB_MAX)
      @inbounds carry[i] = 2
    elseif (res == LIMB_MAX - 1)
      @inbounds carry[i] = 1
    else
      @inbounds carry[i] = 0
    end

    @inbounds out_d[i] = res % LIMB_MAX
    return nothing
  end

  function uadd_carry_kernel(out_d, carry)
    i = idx()
    if (i * 2 > length(carry))
      return nothing
    end

    if (carry[i] == 2)
      @inbounds out_d[i+1] = (out_d[i+1] + 1) % LIMB_MAX
    end
    return nothing
  end

  function uadd(a::IntGPU, b::IntGPU, out::IntGPU)
    nmax = max(length(a), length(b))
    carry = CuEmpty(LIMB_TYPE, nmax*2)

    @cuda blocks=getBlocks(nmax) threads=NUM_THREADS uadd_sum_kernel(a.d, b.d, out.d, carry)

    @cuda blocks=getBlocks(nmax) threads=NUM_THREADS prefix_sum_kernel!(carry, oplus)
    
    @cuda blocks=getBlocks(nmax) threads=NUM_THREADS uadd_carry_kernel(out.d, carry)

    return out
  end

  function +(x::IntGPU, y::IntGPU)
    @assert x.sign == y.sign # Until `usub` and `compare` are implemented
    n = max(length(x), length(y))
    out = IntGPU(true, CuZeros(n + 1))
    out = uadd(x, y, out)
    return out 
  end

  function toBig(x::IntGPU)
    out = BigInt(0.0)
    tmp = Array(x.d)
    for i=1:length(x)
      out += BigInt(tmp[i]) * BigInt(LIMB_MAX) ^ BigInt(i - 1)
    end
    return x.sign ? out : -out
  end

  function fromBig(x::BigInt)
    n = ceil(Int, (Base.GMP.BITS_PER_LIMB * x.alloc) / floor(Int, log2(LIMB_MAX)))
    tmp = Array{LIMB_TYPE}(undef, n)
    for i=1:n
      tmp[i] = x % LIMB_MAX
      x = (x - x % LIMB_MAX) ÷ LIMB_MAX
    end
    out = IntGPU(Bool(x > 0),  CuArray(tmp))
    return out
  end

end

