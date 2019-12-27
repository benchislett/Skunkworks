module CameraDef

  using ..RayDef

  using LinearAlgebra

  struct Camera
    origin::Vec3
    bottom_left::Vec3
    horizontal::Vec3
    vertical::Vec3
    u::Vec3
    v::Vec3
    w::Vec3
    lens_radius::Float32
  end

  function random_in_unit_disk()
    p = Vec3(5, 5, 0)
    while norm(p) >= 1.0f0
      p = 2 * Vec3(rand(), rand(), 0) - Vec3(1, 1, 0)
    end
    return p
  end

  function Camera(look_from::Vec3, look_at::Vec3, view_up::Vec3, fov_vertical::Float32, aspect::Float32, aperture::Float32, focus_dist::Float32)
    lens_radius = aperture / 2
    origin = look_from

    half_height = tan(fov_vertical / 2)
    half_width = aspect * half_height
    
    w = normalize(look_from - look_at)
    u = normalize(cross(view_up, w))
    v = cross(w, u)

    bottom_left = origin - focus_dist * (half_width * u + half_height * v + w)
    horizontal = 2 * focus_dist * half_width * u
    vertical = 2 * focus_dist * half_height * v

    return Camera(origin, bottom_left, horizontal, vertical, u, v, w, lens_radius)
  end

  function get_ray(c::Camera, u::Float32, v::Float32)
    rd = c.lens_radius * random_in_unit_disk()
    offset = c.u * rd[1] + c.v * rd[2]
    return Ray(c.origin .+ offset, c.bottom_left .+ (u .* c.horizontal) .+ (v .* c.vertical) - c.origin .- offset)
  end

  export Camera
  export get_ray
end
