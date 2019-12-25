module RayOps
  using ..RayDef

  const white = Vec3(1.0, 1.0, 1.0)
  const blue = Vec3(0.5, 0.7, 1.0)

  using LinearAlgebra

  function background(v::Vec3)
    y = normalize(v)[2]
    t = 0.5 * (y + 1)
    return ((1 - t) .* white) .+ (t .* blue)
  end

  function get_color(r::Ray)
    return background(r.to)
  end

  export get_color
end
