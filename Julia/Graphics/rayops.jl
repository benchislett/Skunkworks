module RayOps
  using ..RayDef
  using ..Objects

  using IntervalSets

  const white = Vec3(1.0, 1.0, 1.0)
  const blue = Vec3(0.5, 0.7, 1.0)

  using LinearAlgebra

  function background(v::Vec3)
    y = normalize(v)[2]
    t = 0.5 * (y + 1)
    return ((1 - t) .* white) .+ (t .* blue)
  end

  function get_color(r::Ray, world::ObjectSet)
    did_hit, record = hit(r, world, ClosedInterval{Float32}(0.0f0, Inf32))
    if did_hit
      return 0.5 * (record.normal .+ 1)
    else
      return background(r.to)
    end
  end

  export get_color
end
