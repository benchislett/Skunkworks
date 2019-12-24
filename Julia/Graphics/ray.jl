module RayOps
  using LinearAlgebra

  const white = Vector{Float32}([1.0, 1.0, 1.0])
  const blue = Vector{Float32}([0.5, 0.7, 1.0])

  struct ray
    from::Vector{Float32}
    to::Vector{Float32}
  end

  function ray_at(r::ray, t::Float32)
    return r.from .+ t .* r.to
  end

  function background(v::Vector{Float32})
    y = normalize(v)[2]
    t = 0.5 * (y + 1)
    return ((1 - t) .* white) .+ (t .* blue)
  end

  function get_color(r::ray)
    background(r.to)
  end

  export ray, ray_at
  export get_color
end
