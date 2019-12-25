include("ray.jl")
using .RayDef

include("object.jl")
using .Objects

include("rayops.jl")
using .RayOps

using Images

function main()
  w, h = 200, 100
  img = zeros(Float32, 3, h, w)

  origin = Vec3(0, 0, 0)
  window_bottom_left = Vec3(-2, -1, -1)
  window_width = Vec3(4, 0, 0)
  window_height = Vec3(0, 2, 0)

  sphere2 = Sphere(Vec3(0, -100.5, -1), 100)
  sphere1 = Sphere(Vec3(0, 0, -1), 0.5)
  
  world = ObjectSet([sphere1, sphere2])

  for j in h:-1:1
    for i in 1:w
      u = i / w
      v = j / h
      r = Ray(origin, window_bottom_left .+ (u .* window_width) .+ (v .* window_height))
      img[:, h - j + 1, i] = get_color(r, world)
    end
  end

  save("output/output.png", colorview(RGB, img))
end

main()
