module Objects
  using ..RayDef

  using IntervalSets
  using LinearAlgebra

  mutable struct HitRecord
    time::Float32
    point::Vec3
    normal::Vec3
  end

  HitRecord() = HitRecord(0, Vec3(0, 0, 0), Vec3(0, 0, 0))

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

  ObjectSet = Set{Object}

  function hit(r::Ray, o::ObjectSet, t::ClosedInterval{Float32})
    record = HitRecord()
    any_hits = false
    closest = t.right
    for obj in o
      did_hit, temp_record = hit(r, obj, ClosedInterval{Float32}(t.left, closest))
      if did_hit
        any_hits = true
        record = temp_record
        closest = record.time
      end
    end
    return any_hits, record
  end

  export Object, ObjectSet, Sphere
  export HitRecord, hit
end
