module Textures

  using ..RayDef

  abstract type Texture end

  struct ConstantTexture <: Texture
    color::Vec3
  end

  function value(tex::ConstantTexture, u::Float32, v::Float32, p::Vec3)
    return tex.color
  end

  struct CheckerTexture <: Texture
    color1::Texture
    color2::Texture
  end

  function value(tex::CheckerTexture, u::Float32, v::Float32, p::Vec3)
    sines = prod(sin.(10 .* p))
    if sines < 0
      return value(tex.color1, u, v, p)
    else
      return value(tex.color2, u, v, p)
    end
  end

  struct ImageTexture <: Texture
    data::Array{Float32, 3} # RGB, h, w
  end

  function value(tex::ImageTexture, u::Float32, v::Float32, p::Vec3)
    h, w = size(tex.data)[2:3]
    i = floor(Int, clamp(u * w + 1, 1, w))
    j = floor(Int, clamp((1 - v) * h + 0.999, 1, h))
    
    val = tex.data[:, j, i]
    
    return Vec3(val[1], val[2], val[3])
  end

  export Texture
  export ConstantTexture, CheckerTexture, ImageTexture
  export value
end
