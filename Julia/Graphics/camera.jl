module CameraDef

  using ..RayDef

  using LinearAlgebra

  struct Camera
    origin::Vec3
    bottom_left::Vec3
    horizontal::Vec3
    vertical::Vec3
  end

  function Camera(look_from::Vec3, look_at::Vec3, view_up::Vec3, fov_vertical::Float32, aspect::Float32)
    origin = look_from

    half_height = tan(fov_vertical / 2)
    half_width = aspect * half_height
    
    w = normalize(look_from - look_at)
    u = normalize(cross(view_up, w))
    v = cross(w, u)

    bottom_left = origin - half_width * u - half_height * v - w
    horizontal = 2 * half_width * u
    vertical = 2 * half_height * v

    return Camera(origin, bottom_left, horizontal, vertical)
  end

  function get_ray(c::Camera, u::Float32, v::Float32)
    return Ray(c.origin, c.bottom_left .+ (u .* c.horizontal) .+ (v .* c.vertical) - c.origin)
  end

  export Camera
  export get_ray
end
