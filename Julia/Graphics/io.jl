module FileIO

  using ..RayDef
  using ..Objects
  using ..Materials

  function loadImage(path::String)
    return convert(Array{Float32, 3}, channelview(load(path)))
  end

  function loadOFF(path::String, world::ObjectSet, mat::Material)
    io = open(path, "r")
    @assert readline(io) == "OFF"

    n_verts, n_faces, n_edges = map(x->parse(Int,x), split(readline(io)))

    vertices = Array{Vec3, 1}(undef, n_verts)
    for i in 1:n_verts
      vertices[i] = Vec3(map(x->parse(Float32,x), split(readline(io)))...)
    end

    for i in 1:n_faces
      line = map(x->parse(Int,x), split(readline(io)))
      if line[1] == 3
        push!(world, Tri(vertices[line[2]+1], vertices[line[3]+1], vertices[line[4]+1], mat))
      elseif line[1] == 4
        push!(world, Quad(vertices[line[2]+1], vertices[line[3]+1], vertices[line[4]+1], vertices[line[5]+1], mat))
      end
    end
  end

  export loadImage, loadOFF
end
