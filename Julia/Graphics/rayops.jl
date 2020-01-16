module RayOps
  using ..RayDef
  using ..Materials
  using ..Objects

  using IntervalSets

  const white = Vec3(1.0, 1.0, 1.0)
  const blue = Vec3(0.5, 0.7, 1.0)

  using LinearAlgebra

  function background(v::Vec3)
    return Vec3(0.7, 0.7, 0.7)
    # y = normalize(v)[2]
    # t = 0.5 * (y + 1)
    # return ((1 - t) .* white) .+ (t .* blue)
  end

  function get_color(r::Ray, world, depth::Int = 1)
    did_hit, record = hit(r, world, OpenInterval{Float32}(0.001f0, Inf32))
    if did_hit
      emittance = emitted(record.material, record.u, record.v, record.point)
      did_scatter, scattered_ray, attenuation = scatter(record.material, r, record)
      if depth < 50 && did_scatter
        return emittance .+ (attenuation .* get_color(scattered_ray, world, depth + 1))
      else
        return emittance
      end
    else
      return background(r.to)
    end
  end

  export get_color
end
