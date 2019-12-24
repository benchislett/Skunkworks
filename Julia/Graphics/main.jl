include("ray.jl")
using .RayOps

using Images

function main()
  w, h = 200, 100
  img = zeros(Float32, 3, h, w)

  origin = Vec3([0, 0, 0])
  window_bottom_left = Vec3([2, -1, -1])
  window_width = Vec3([4, 0, 0])
  window_height = Vec3([0, 2, 0])

  for j in 1:h
    for i in 1:w
      u = i / w
      v = j / h
      r = ray(origin, window_bottom_left .+ (u .* window_width) .+ (v .* window_height))
      img[:, j, i] = get_color(r)
    end
  end

  save("./output.png", colorview(RGB, img))
end

main()
