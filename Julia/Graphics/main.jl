include("ray.jl")
using .RayDef

include("object.jl")
using .Objects

include("rayops.jl")
using .RayOps

include("camera.jl")
using .CameraDef

using Images

function main()
  w, h = 200, 100
  samples = 250
  img = zeros(Float32, 3, h, w)

  cam = Camera()

  sphere1 = Sphere(Vec3(0, -100.5, -1), 100)
  sphere2 = Sphere(Vec3(0, 0, -1), 0.5)
  
  world = ObjectSet([sphere1, sphere2])

  for j in h:-1:1
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
