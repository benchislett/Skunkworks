module Materials

  using ..RayDef

  using LinearAlgebra

  abstract type Material end

  mutable struct HitRecord
    time::Float32
    point::Vec3
    normal::Vec3
    material::Material
  end

  HitRecord() = HitRecord(0, Vec3(0, 0, 0), Vec3(0, 0, 0), Diffuse(Vec3(0.8, 0.8, 0.8)))

  function random_unit_sphere()
    p = Vec3(5, 5, 5)
    while norm(p) >= 1
      p = 2 .* Vec3(rand(), rand(), rand()) .- 1
    end
    return p
  end

  struct Diffuse <: Material
    albedo::Vec3
  end

  function scatter(mat::Diffuse, r::Ray, record::HitRecord)
    random_target = record.point + record.normal + random_unit_sphere()
    scattered = Ray(record.point, random_target - record.point)
    attenuation = mat.albedo
    return (true, scattered, attenuation)
  end

  struct Metal <: Material
    albedo::Vec3
    fuzz::Float32
    Metal(a::Vec3, f::Float32) = new(a, min(1.0f0, f))
  end

  function reflect(v::Vec3, n::Vec3)
    return v - 2 * dot(v, n) * n
  end

  function scatter(mat::Metal, r::Ray, record::HitRecord)
    reflected = reflect(normalize(r.to), record.normal)
    scattered = Ray(record.point, reflected + mat.fuzz * random_unit_sphere())
    attenuation = mat.albedo
    did_scatter = dot(scattered.to, record.normal) > 0
    return (did_scatter, scattered, attenuation)
  end

  function refract(v::Vec3, n::Vec3, ni_over_nt::Float32)
    unit_v = normalize(v)
    dt = dot(unit_v, n)
    discriminant = 1.0f0 - (ni_over_nt*ni_over_nt) * (1 - (dt*dt))
    if (discriminant > 0)
      refracted = ni_over_nt * (unit_v - n * dt) - n * sqrt(discriminant)
      return (true, refracted)
    else
      return (false, Vec3(0, 0, 0))
    end
  end

  function schlick(cos::Float32, refraction_index::Float32)
    r0 = (1 - refraction_index) / (1 + refraction_index)
    r0 = r0 * r0
    return r0 + (1 - r0) * ((1 - cos)^5)
  end

  struct Dielectric <: Material
    refraction_index::Float32
  end

  function scatter(mat::Dielectric, r::Ray, record::HitRecord)
    reflected = reflect(r.to, record.normal)
    attenuation = Vec3(1, 1, 1)
    
    if dot(r.to, record.normal) > 0
      outward_normal = -record.normal
      ni_over_nt = mat.refraction_index
      cos = mat.refraction_index * dot(r.to, record.normal) / norm(r.to)
    else
      outward_normal = record.normal
      ni_over_nt = 1.0f0 / mat.refraction_index
      cos = -dot(r.to, record.normal) / norm(r.to)
    end

    did_refract, refracted = refract(r.to, outward_normal, ni_over_nt)
    if did_refract
      reflect_prob = schlick(cos, mat.refraction_index)
    else
      reflect_prob = 1.0f0
    end

    if rand() < reflect_prob
      scattered = Ray(record.point, reflected)
    else
      scattered = Ray(record.point, refracted)
    end

    return (true, scattered, attenuation)
  end

  export HitRecord
  export Material, Diffuse, Metal, Dielectric
  export scatter
end

