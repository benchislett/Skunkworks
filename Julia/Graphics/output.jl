include("ray.jl")
using .RayOps

using Images

function main()
  w, h = 200, 100
  img = zeros(Float32, 3, h, w)

  origin = Vector{Float32}([0, 0, 0])
  window_bottom_left = Vector{Float32}([-2, -1, -1])
  window_width = Vector{Float32}([4, 0, 0])
  window_height = Vector{Float32}([0, 2, 0])

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
