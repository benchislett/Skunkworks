module CameraDef

  using ..RayDef

  origin = Vec3(0, 0, 0)

  struct Camera
    bottom_left::Vec3
    horizontal::Vec3
    vertical::Vec3
  end

  Camera() = Camera(Vec3(-2, -1, -1), Vec3(4, 0, 0), Vec3(0, 2, 0))

  function get_ray(c::Camera, u::Float32, v::Float32)
    return Ray(origin, c.bottom_left .+ (u .* c.horizontal) .+ (v .* c.vertical))
  end

  export Camera
  export get_ray
end
