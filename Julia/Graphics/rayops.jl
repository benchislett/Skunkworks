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

  function random_unit_sphere()
    p = Vec3(5, 5, 5)
    while norm(p) >= 1
      p = 2 .* Vec3(rand(), rand(), rand()) .- 1
    end
    return p
  end

  function get_color(r::Ray, world::ObjectSet)
    did_hit, record = hit(r, world, ClosedInterval{Float32}(0.01f0, Inf32))
    if did_hit
      rand_target = record.point + record.normal + random_unit_sphere()
      return 0.5 * get_color(Ray(record.point, rand_target - record.point), world)
    else
      return background(r.to)
    end
  end

  export get_color
end
