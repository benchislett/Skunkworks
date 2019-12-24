module RayOps
  using LinearAlgebra
  using StaticArrays

  Vec3 = SVector{3, Float32}
  Vec3() = Vec3([0.0f0, 0.0f0, 0.0f0])

  const white = Vec3([1.0, 1.0, 1.0])
  const blue = Vec3([0.5, 0.7, 1.0])

  struct Ray
    from::Vec3
    to::Vec3
  end

  function ray_at(r::Ray, t::Float32)
    return r.from .+ t .* r.to
  end

  function background(v::Vec3)
    y = normalize(v)[2]
    t = 0.5 * (y + 1)
    return ((1 - t) .* white) .+ (t .* blue)
  end

  function get_color(r::Ray)
    background(r.to)
  end

  export Vec3
  export Ray, ray_at
  export get_color
end
