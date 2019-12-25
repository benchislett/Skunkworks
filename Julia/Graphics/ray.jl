module RayDef

  using LinearAlgebra
  using StaticArrays

  Vec3 = SVector{3, Float32}

  struct Ray
    from::Vec3
    to::Vec3
  end

  function ray_at(r::Ray, t::Float32)
    return r.from .+ t .* r.to
  end

  export Vec3
  export Ray, ray_at
end
