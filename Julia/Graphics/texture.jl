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

  export Texture
  export ConstantTexture, CheckerTexture
  export value
end
