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
  w, h = 100, 100
  samples = 50
  img = zeros(Float32, 3, h, w)

  camera_pos = Vec3(278, 278, -800)
  camera_target = Vec3(278, 278, 0)
  focus_dist = norm(camera_pos - camera_target)
  aperture = 0.0f0
  fov = Float32(2 * pi / 9)

  cam = Camera(camera_pos, camera_target, Vec3(0, 1, 0), fov, Float32(w / h), aperture, focus_dist)

  world = ObjectSet()
  
  red = Diffuse(ConstantTexture(Vec3(0.65, 0.05, 0.05)))
  white = Diffuse(ConstantTexture(Vec3(0.73, 0.73, 0.73)))
  green = Diffuse(ConstantTexture(Vec3(0.12, 0.45, 0.15)))
  light = Light(ConstantTexture(Vec3(14, 14, 14)))

  push!(world, TrueRect(Vec3(555, 0, 0), Vec3(555, 555, 0), Vec3(555, 555, 555), green))
  push!(world, TrueRect(Vec3(0, 0, 0), Vec3(0, 555, 0), Vec3(0, 555, 555), red))
  push!(world, TrueRect(Vec3(213, 554, 227), Vec3(213, 554, 332), Vec3(343, 554, 332), light))
  #push!(world, TrueRect(Vec3(0, 555, 0), Vec3(555, 555, 0), Vec3(555, 555, 555), white))
  #push!(world, TrueRect(Vec3(0, 0, 0), Vec3(555, 0, 0), Vec3(555, 0, 555), white))
  #push!(world, TrueRect(Vec3(0, 0, 555), Vec3(0, 555, 555), Vec3(555, 555, 555), white))

  white_wall_points = Matrix{Vec3}(undef, (4, 2))
  white_wall_points[1, 1] = Vec3(0, 0, 0)
  white_wall_points[1, 2] = Vec3(555, 0, 0)
  white_wall_points[2, 1] = Vec3(0, 0, 555)
  white_wall_points[2, 2] = Vec3(555, 0, 555)
  white_wall_points[3, 1] = Vec3(0, 555, 555)
  white_wall_points[3, 2] = Vec3(555, 555, 555)
  white_wall_points[4, 1] = Vec3(0, 555, 0)
  white_wall_points[4, 2] = Vec3(555, 555, 0)

  white_wall_patch = Patch(white_wall_points, white)
  push!(world, white_wall_patch)

  world = make_bvh(world)
  
  try
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
  finally
    img = map(clamp01nan, img)
    save("output/output.png", colorview(RGB, img))
  end

end

main()
