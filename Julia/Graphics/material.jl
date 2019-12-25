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

  export HitRecord
  export Material, Diffuse
  export scatter
end
