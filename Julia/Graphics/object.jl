module Objects

  include("ray.jl")
  using .RayOps

  struct HitRecord
    time::Float32
    point::Vec3
    normal::Vec3
  end

  HitRecord() = HitRecord(0.0, Vec3(), Vec3())

  abstract type Object end

  struct Sphere <: Object
    center::Vec3
    radius::Float32
  end

  function hit(r::Ray, s::Sphere, t::ClosedInterval{Float32})
    record = HitRecord()

    ray_to_sphere = r.from - s.center

    a = dot(r.to, r.to)
    b = dot(ray_to_sphere, r.to)
    c = dot(ray_to_sphere, ray_to_sphere) - s.radius^2

    # Cancel 2's
    discriminant = b^2 - a * c
    
    if discriminant < 0
      return false, record
    else
      t1 = (-b - sqrt(discriminant)) / a
      t2 = (-b + sqrt(discriminant)) / a

      if t1 in t || t2 in t
        record.time = min(t1, t2)
        record.point = ray_at(r, record.time)
        record.normal = (record.point - s.center) / s.radius
        return true, record
      else
        return false, record
      end
    end
  end

end
