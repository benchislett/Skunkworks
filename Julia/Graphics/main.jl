include("ray.jl")
using .RayDef

include("texture.jl")
using .Textures

include("material.jl")
using .Materials

include("object.jl")
using .Objects

include("io.jl")
using .FileIO

include("rayops.jl")
using .RayOps

include("camera.jl")
using .CameraDef

using Images
using ProgressMeter
using LinearAlgebra
using Base.Threads

function random_color()
  return Vec3(rand(), rand(), rand())
end

function random_light_color()
  return Vec3(0.5*(1 + rand()), 0.5*(1 + rand()), 0.5*(1 + rand()))
end

function main()
  w, h = 128, 128
  samples = 50
  img = zeros(Float32, 3, h, w)

  world = ObjectSet()
  
  red = Diffuse(ConstantTexture(Vec3(0.65, 0.05, 0.05)))
  white = Diffuse(ConstantTexture(Vec3(0.73, 0.73, 0.73)))
  green = Diffuse(ConstantTexture(Vec3(0.12, 0.45, 0.15)))
  light = Light(ConstantTexture(Vec3(14, 14, 14)))

  br = 0.25f0 # Half Box Side Length
  push!(world, Quad(Vec3(-br, -br, -br), Vec3(-br, -br, br), Vec3(br, -br, br), Vec3(br, -br, -br), white))
  push!(world, Quad(Vec3(-br, -br, br), Vec3(-br, br, br), Vec3(br, br, br), Vec3(br, -br, br), white))
  push!(world, Quad(Vec3(-br, br, -br), Vec3(-br, br, br), Vec3(br, br, br), Vec3(br, br, -br), white))
  push!(world, Quad(Vec3(-br, -br, -br), Vec3(-br, br, -br), Vec3(-br, br, br), Vec3(-br, -br, br), green))
  push!(world, Quad(Vec3(br, -br, -br), Vec3(br, -br, br), Vec3(br, br, br), Vec3(br, br, -br), red))

  push!(world, Quad(Vec3(-br/2, br, -br/2), Vec3(-br/2, br, br/2), Vec3(br/2, br, br/2), Vec3(br/2, br, -br/2), light))

  loadOFF("data/bunny.off", world, white)

  world = make_bvh(world)

  camera_pos = Vec3(0, 0, -br)
  camera_target = Vec3(0, 0, 0)
  focus_dist = norm(camera_pos - camera_target)
  aperture = 0.0f0
  fov = Float32(2π/ 9)

  cam = Camera(camera_pos, camera_target, Vec3(0, 1, 0), fov, Float32(w / h), aperture, focus_dist)

  try
    p = Progress(w * h, 1, "Rendering...")
    @threads for j in h:-1:1
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
  finally
    img = map(clamp01nan, img)
    save("output/output.png", colorview(RGB, img))
  end

end

main()
