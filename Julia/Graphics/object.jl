module Objects
  using ..RayDef
  using ..Materials

  using IntervalSets
  using LinearAlgebra
  using StrLiterals
  using ProgressMeter

  abstract type Object end

  struct Slab <: Object
    lower_left::Vec3
    upper_right::Vec3
  end

  function bounding_slab(s::Slab)
    return s
  end

  function hit(r::Ray, s::Slab, t::OpenInterval{Float32})
    t0 = min.((s.lower_left .- r.from) ./ r.to, (s.upper_right .- r.from) ./ r.to)
    t1 = max.((s.lower_left .- r.from) ./ r.to, (s.upper_right .- r.from) ./ r.to)
    tmin = max.(t0, t.left)
    tmax = min.(t1, t.right)
    return all(tmax .> tmin)
  end

  function superslab(s1::Slab, s2::Slab)
    return Slab(min.(s1.lower_left, s2.lower_left), max.(s1.upper_right, s2.upper_right))
  end

  struct Sphere <: Object
    center::Vec3
    radius::Float32
    material::Material
  end

  function bounding_slab(s::Sphere)
    return Slab(s.center .- s.radius, s.center .+ s.radius)
  end

  function get_sphere_uv(point::Vec3)
    phi = atan(point[3], point[1])
    theta = asin(clamp(point[2], -1, 1))
    u::Float32 = 1 - (phi + pi) / (2 * pi)
    v::Float32 = (theta + pi / 2) / pi
    return u, v
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
        record.u, record.v = get_sphere_uv(record.normal)
        return true, record
      else
        return false, record
      end
    end
  end

  ObjectSet = Array{Object, 1}

  function bounding_slab(o::ObjectSet)
    s = bounding_slab(o[1])
    for obj in o
      s = superslab(s, bounding_slab(obj))
    end
    return s
  end

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

  struct BoundingNode <: Object
    left::Object
    right::Object
    slab::Slab
  end

  function bounding_slab(node::BoundingNode)
    return node.slab
  end

  function hit(r::Ray, node::BoundingNode, t::OpenInterval{Float32})
    if hit(r, node.slab, t)
      did_hit_left, left_record = hit(r, node.left, t)
      did_hit_right, right_record = hit(r, node.right, t)

      if did_hit_left && did_hit_right
        if left_record.time < right_record.time
          return true, left_record
        else
          return true, right_record
        end
      elseif did_hit_left
        return true, left_record
      elseif did_hit_right
        return true, right_record
      end
    end
    return false, HitRecord()
  end

  function sort_along_axis!(world::ObjectSet, axis::Int)
    sort!(world, by=obj->bounding_slab(obj).lower_left[axis])
  end

  function d(x::Object, y::Object) # Cluster distance function
    slab = superslab(bounding_slab(x), bounding_slab(y))
    l, w, h = slab.upper_right .- slab.lower_left
    result = 2 * ((l * w) + (w * h) + (l * h))
    return result
  end

  function make_bvh(world::ObjectSet) # Agglomerative Clustering
    clusters = copy(world)
    p = Progress(length(clusters), 1, "Constructing BVH...")
    while length(clusters) > 2
      next!(p)
      best = Inf32
      left = clusters[1]
      right = clusters[1]
      for obj1 in clusters
        for obj2 in clusters
          diff = d(obj1, obj2)
          if obj1 != obj2 && diff < best
            left = obj1
            right = obj2
            best = diff
          end
        end
      end
      deleteat!(clusters, findfirst(x->x==left, clusters))
      deleteat!(clusters, findfirst(x->x==right, clusters))
      push!(clusters, BoundingNode(left, right, superslab(bounding_slab(left), bounding_slab(right))))
    end

    return BoundingNode(clusters[1], clusters[2], superslab(bounding_slab(clusters[1]), bounding_slab(clusters[1])))
  end

  struct AxisRect <: Object
    lower_left::Vec3
    upper_right::Vec3
    material::Material
    axis::Int
    AxisRect(ll::Vec3, ur::Vec3, mat::Material) = new(ll, ur, mat, findfirst(iszero, ll .- ur))
  end

  function single_one(index::Int)
    arr = Matrix(1.0f0I, 3, 3)[index, :]
    return Vec3(arr...)
  end

  function hit(r::Ray, rect::AxisRect, t::OpenInterval{Float32})
    record = HitRecord()

    axis_depth = rect.lower_left[rect.axis]
    time = (axis_depth - r.from[rect.axis]) / r.to[rect.axis]
    
    if time in t
      point = ray_at(r, time)
      if all(rect.upper_right .>= point .>= rect.lower_left)
        uv_xyz_points = collect((point .-  rect.lower_left) ./ (rect.upper_right .- rect.lower_left))
        deleteat!(uv_xyz_points, rect.axis)
        record.u, record.v = uv_xyz_points
        record.time = time
        record.material = rect.material
        record.point = point
        if r.from[rect.axis] > rect.lower_left[rect.axis]
          record.normal = single_one(rect.axis)
        else
          record.normal = -single_one(rect.axis)
        end
        return true, record
      end
    end

    return false, record
  end

  function bounding_slab(rect::AxisRect)
    one_axis = single_one(rect.axis)
    return Slab(rect.lower_left .- 0.001 .* one_axis, rect.upper_right .+ 0.001 .* one_axis)
  end

  struct Tri <: Object
    a::Vec3
    b::Vec3
    c::Vec3
    u::Vec3
    v::Vec3
    normal::Vec3
    material::Material
    slab::Slab
  end

  function Tri(a::Vec3, b::Vec3, c::Vec3, mat::Material)
    u = b .- a
    v = c .- a

    normal = normalize(cross(u, v))
    slab = Slab(min.(a, b, c) .- 0.001, max.(a, b, c) .+ 0.001)
    return Tri(a, b, c, u, v, normal, mat, slab)
  end

  function hit(r::Ray, tri::Tri, t::OpenInterval{Float32}) # Moller-Trumbore
    record = HitRecord()

    h = cross(r.to, tri.v)
    a = dot(tri.u, h)

    if iszero(a) return false, record end
    
    f = 1.0f0 / a
    s = r.from - tri.a
    u = f * dot(s, h)
    
    if !(0.0f0 <= u <= 1.0f0) return false, record end

    q = cross(s, tri.u)
    v = f * dot(r.to, q)

    if (v < 0.0f0 || u + v > 1.0f0) return false, record end

    time = f * dot(tri.v, q)
    if time in t
      record.point = ray_at(r, time)
      record.time = time
      record.material = tri.material
      record.u, record.v = u, v
      record.normal = -sign(dot(r.to, tri.normal)) * tri.normal
      return true, record
    else
      return false, record
    end
  end

  function bounding_slab(tri::Tri)
    return tri.slab
  end

  struct Quad <: Object
    tri1::Tri
    tri2::Tri
  end

  function Quad(a::Vec3, b::Vec3, c::Vec3, d::Vec3, mat::Material)
    tri1 = Tri(a, b, c, mat)
    tri2 = Tri(a, c, d, mat)
    return Quad(tri1, tri2)
  end

  function hit(r::Ray, q::Quad, t::OpenInterval{Float32})
    hit1, record1 = hit(r, q.tri1, t)
    hit2, record2 = hit(r, q.tri2, t)

    if (!hit1 && !hit2) return false, HitRecord() end
    if (!hit1) return true, record2 end
    if (!hit2) return true, record1 end

    if record1.time < record2.time
      return true, record1
    else
      return true, record2
    end
  end

  function bounding_slab(q::Quad)
    return superslab(bounding_slab(q.tri1), bounding_slab(q.tri2))
  end

  struct Box <: Object
    lower_left::Vec3
    upper_right::Vec3
    rects::ObjectSet
  end

  function Box(lower_left::Vec3, upper_right::Vec3, mat::Material)
    rects = ObjectSet()
    push!(rects, AxisRect(lower_left, Vec3(upper_right[1], upper_right[2], lower_left[3]), mat))
    push!(rects, AxisRect(Vec3(lower_left[1], lower_left[2], upper_right[3]), upper_right, mat))
    push!(rects, AxisRect(lower_left, Vec3(upper_right[1], lower_left[2], upper_right[3]), mat))
    push!(rects, AxisRect(Vec3(lower_left[1], upper_right[2], lower_left[3]), upper_right, mat))
    push!(rects, AxisRect(lower_left, Vec3(lower_left[1], upper_right[2], upper_right[3]), mat))
    push!(rects, AxisRect(Vec3(upper_right[1], lower_left[2], lower_left[3]), upper_right, mat))
    return Box(lower_left, upper_right, rects)
  end

  function hit(r::Ray, b::Box, t::OpenInterval{Float32})
    return hit(r, b.rects, t)
  end

  function bounding_slab(b::Box)
    return Slab(b.lower_left, b.upper_right)
  end
  
  export Object, ObjectSet, Sphere, Slab, BoundingNode, AxisRect, Box, Tri, Quad
  export hit, make_bvh
end
