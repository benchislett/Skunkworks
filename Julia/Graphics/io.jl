module IO

  function loadImage(path::String)
    return convert(Array{Float32, 3}, channelview(load(path)))
  end

  export loadImage
end
