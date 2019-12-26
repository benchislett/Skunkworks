module Objects
  using ..RayDef
  using ..Materials

  using IntervalSets
  using LinearAlgebra

  abstract type Object end

  struct Sphere <: Object
    center::Vec3
    radius::Float32
    material::Material
  end

  function hit(r::Ray, s::Sphere, t::OpenInterval{Float32})
    record = HitRecord()

    ray_to_sphere = r.from - s.center

    a = dot(r.to, r.to)
    b = dot(ray_to_sphere, r.to)
    c = dot(ray_to_sphere, ray_to_sphere) - s.radius * s.radius

    # Cancel 2's
    discriminant = b * b - a * c
    
    if discriminant <= 0
      return false, record
    else
      t1 = (-b - sqrt(discriminant)) / a
      t2 = (-b + sqrt(discriminant)) / a

      if t1 in t || t2 in t
        if t1 in t
          record.time = t1
        else
          record.time = t2
        end
        record.point = ray_at(r, record.time)
        record.normal = (record.point - s.center) / s.radius
        record.material = s.material
        return true, record
      else
        return false, record
      end
    end
  end

  ObjectSet = Set{Object}

  function hit(r::Ray, o::ObjectSet, t::OpenInterval{Float32})
    record = HitRecord()
    any_hits = false
    closest = t.right
    for obj in o
      did_hit, temp_record = hit(r, obj, OpenInterval{Float32}(t.left, closest))
      if did_hit
        any_hits = true
        record = temp_record
        closest = record.time
      end
    end
    return any_hits, record
  end

  export Object, ObjectSet, Sphere
  export hit
end
