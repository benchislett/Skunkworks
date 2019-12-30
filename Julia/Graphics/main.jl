include("ray.jl")
using .RayDef

include("texture.jl")
using .Textures

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
  return Vec3(rand(), rand(), rand())
end

function random_light_color()
  return Vec3(0.5*(1 + rand()), 0.5*(1 + rand()), 0.5*(1 + rand()))
end

function main()
  w, h = 200, 100
  samples = 50
  img = zeros(Float32, 3, h, w)

  camera_pos = Vec3(8, 0, 2)
  camera_target = Vec3(0, 0, 0)
  focus_dist = norm(camera_pos - camera_target)
  aperture = 0.1f0

  cam = Camera(camera_pos, camera_target, Vec3(0, 1, 0), Float32(pi/6), Float32(w / h), aperture, focus_dist)

  world = ObjectSet()
  earth_img = load("data/earthmap.png")
  earth_img = convert(Array{Float32, 3}, channelview(earth_img))
  push!(world, Sphere(Vec3(0, 0, 0), 2.0f0, Diffuse(ImageTexture(earth_img))))
  world = make_bvh(world)

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
