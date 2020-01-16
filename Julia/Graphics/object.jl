module Objects
  using ..RayDef
  using ..Materials

  using IntervalSets
  using LinearAlgebra

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

  function make_bvh(world::ObjectSet)
    axis = rand([1,2,3])
    sort_along_axis!(world, axis)
    
    n = length(world)
    if n == 1
      left = right = world[1]
    elseif n == 2
      left, right = world
    else
      left = make_bvh(world[1:floor(Int, n/2)])
      right = make_bvh(world[floor(Int, n/2) + 1:n])
    end
    
    return BoundingNode(left, right, superslab(bounding_slab(left), bounding_slab(right)))
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

  struct TrueRect <: Object
    corner::Vec3
    u::Vec3
    v::Vec3
    normal::Vec3
    change_of_basis_matrix::Matrix{Float32}
    material::Material
  end

  function TrueRect(a::Vec3, mid::Vec3, c::Vec3, mat::Material) # 3 Points
    u = a .- mid
    v = c .- mid
    normal = normalize(cross(u, v))
    matrix = inv([u v normal])
    return TrueRect(mid, u, v, normal, matrix, mat)
  end

  function hit(r::Ray, rect::TrueRect, t::OpenInterval{Float32})
    record = HitRecord()

    slope_dot_normal = dot(r.to, rect.normal)

    if iszero(slope_dot_normal)
      return false, record
    else
      time = dot(rect.corner .- r.from, rect.normal) / slope_dot_normal
      if time in t
        point = ray_at(r, time)
        u, v, _ = rect.change_of_basis_matrix * (point .- rect.corner)
        if 0 <= u <= 1 && 0 <= v <= 1
          record.point = point
          record.time = time
          record.material = rect.material
          record.u, record.v = u, v
          record.normal = -sign(slope_dot_normal) * rect.normal
          return true, record
        else
          return false, record
        end
      else
        return false, record
      end
    end
  end

  function bounding_slab(rect::TrueRect)
    p1 = rect.corner
    p2 = rect.corner .+ rect.u
    p3 = rect.corner .+ rect.v
    p4 = rect.corner .+ rect.u .+ rect.v
    return Slab(min.(p1, p2, p3, p4) .- 0.001, max.(p1, p2, p3, p4) .+ 0.001)
  end

  struct Patch <: Object
    points::Array{Vec3, 2}
    rects::ObjectSet
  end

  function Patch(points::Array{Vec3, 2}, mat::Material)
    rects = ObjectSet()
    for j in 1:(size(points, 2)-1)
      for i in 1:(size(points, 1)-1)
        push!(rects, TrueRect(points[i, j], points[i, j + 1], points[i + 1, j + 1], mat))
      end
    end
    return Patch(points, rects)
  end

  function hit(r::Ray, p::Patch, t::OpenInterval{Float32})
    return hit(r, p.rects, t)
  end

  function bounding_slab(p::Patch)
    return bounding_slab(p.rects)
  end

  struct PatchSet <: Object
    patches::ObjectSet
    PatchSet() = new(ObjectSet())
  end

  function hit(r::Ray, p::PatchSet, t::OpenInterval{Float32})
    return hit(r, p.patches, t)
  end

  function bounding_slab(p::PatchSet)
    return bounding_slab(p.patches)
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
  
  export Object, ObjectSet, Sphere, Slab, BoundingNode, AxisRect, Box, TrueRect, Patch, PatchSet
  export hit, make_bvh
end
