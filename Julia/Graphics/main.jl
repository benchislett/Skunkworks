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

function main()
  w, h = 400, 200
  samples = 500
  img = zeros(Float32, 3, h, w)

  cam = Camera(Vec3(-2, 2, 1), Vec3(0, -0.175, -1), Vec3(0, 1, 0), Float32(pi/5.75), Float32(w / h))

  sphere1 = Sphere(Vec3(0, 0, -1), 0.5, Diffuse(Vec3(0.1, 0.2, 0.5)))
  sphere2 = Sphere(Vec3(0, -100.5, -1), 100, Diffuse(Vec3(0.8, 0.8, 0.0)))
  sphere3 = Sphere(Vec3(1, 0, -1), 0.5, Metal(Vec3(0.8, 0.6, 0.2), 0.3f0))
  sphere4 = Sphere(Vec3(-1, 0, -1), 0.5, Dielectric(1.5f0))
  sphere5 = Sphere(Vec3(-1, 0, -1), -0.45, Dielectric(1.5f0))
  
  world = ObjectSet([sphere1, sphere2, sphere3, sphere4, sphere5])

  @showprogress 1 "Rendering..." for j in h:-1:1
    for i in 1:w
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
