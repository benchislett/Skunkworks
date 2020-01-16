module FileIO

  using ..RayDef
  using ..Objects
  using ..Materials

  function loadImage(path::String)
    return convert(Array{Float32, 3}, channelview(load(path)))
  end

  function loadPoints(io::IOStream)
    u, v = map(x->parse(Int, x), split(readline(io)))
    points = Matrix{Vec3}(undef, (u+1, v+1))
    for j in 1:u+1
      for k in 1:v+1
        points[j, k] = Vec3(map(x->parse(Float32,x), split(readline(io)))...)
      end
    end
    return points
  end

  function loadPatches(path::String, scale::Float32, pos::Vec3, mat::Material)
    io = open(path, "r")
    num_patches = parse(Int, readline(io))
    patches = PatchSet()
    for i in 1:num_patches
      points = map(point -> (point .* scale) .+ pos, loadPoints(io))
      push!(patches.patches, Patch(points, mat))
    end
    return patches
  end

  export loadImage, loadPatches
end
