include("ray.jl")
using .RayDef

include("material.jl")
using .Materials

include("object.jl")
using .Objects

include("rayops.jl")
using .RayOps

include("camera.jl")
using .CameraDef

using Images
using ProgressMeter
using LinearAlgebra

function random_color()
  return Vec3(0.5*(1 + rand()), 0.5*(1 + rand()), 0.5*(1 + rand()))
end

function random_world()
  world = ObjectSet()
  
  # Surface
  push!(world, Sphere(Vec3(0, -1000, 0), 1000, Diffuse(Vec3(0.5, 0.5, 0.5))))
  
  for x in -11:11
    for z in -11:11
      center = Vec3(x + 0.9*rand(), 0.2, z + 0.9*rand())

      if norm(center - Vec3(4, 0.2, 0)) > 0.9
        mat_choice = rand()
        if mat_choice < 0.6 # Diffuse
          push!(world, Sphere(center, 0.2, Diffuse(random_color())))
        elseif mat_choice < 0.8 # Metal
          push!(world, Sphere(center, 0.2, Metal(random_color(), 0.3f0*rand(Float32))))
        else # Glass
          push!(world, Sphere(center, 0.2, Dielectric(1.5f0)))
        end
      end
    end
  end

  # Central Spheres
  push!(world, Sphere(Vec3(0, 1, 0), 1.0f0, Dielectric(1.5f0)))
  push!(world, Sphere(Vec3(-4, 1, 0), 1.0f0, Diffuse(Vec3(0.4, 0.2, 0.1))))
  push!(world, Sphere(Vec3(4, 1, 0), 1.0f0, Metal(Vec3(0.7, 0.6, 0.5), 0.0f0)))

  return world
end


function main()
  w, h = 400, 200
  samples = 250
  img = zeros(Float32, 3, h, w)

  camera_pos = Vec3(13, 2, 3)
  camera_target = Vec3(0, 0, 0)
  focus_dist = norm(camera_pos - camera_target)
  aperture = 0.1f0

  cam = Camera(camera_pos, camera_target, Vec3(0, 1, 0), Float32(pi/6), Float32(w / h), aperture, focus_dist)

  world = random_world()

  p = Progress(w * h, 1, "Rendering...")
  for j in h:-1:1
    for i in 1:w
      next!(p)
      color = Vec3(0, 0, 0)
      for _ in 1:samples
        u::Float32 = (i + rand()) / w
        v::Float32 = (j + rand()) / h

        r = get_ray(cam, u, v)
        c = get_color(r, world)
        color += c
      end
      img[:, h - j + 1, i] = sqrt.(color / samples)
    end
  end

  save("output/output.png", colorview(RGB, img))
end

main()
