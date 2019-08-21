mutable struct State
  field::Array{Complex{Float64}, 2}
  fieldIterations::Array{Int64, 2}

  iterations::Int64

  res::Tuple{Float64, Float64}

  bounds::Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}
  lengths::Tuple{Float64, Float64}
  
  delta::Tuple{Float64, Float64}
  deltas::Array{Complex{Float64}, 2}

  op::Function
end

function State(iterations=20.0, res=(512, 512), bounds=((-2, 2), (-2, 2)), op=(z,c)->z^2+c)
  field = zeros(Complex{Float64}, res)
  fieldIterations = zeros(Int64, res)

  lengths = map(b -> b[2] - b[1], bounds)

  delta = lengths ./ res
  deltas = map(Idx -> complex((Idx.I .* delta)...), CartesianIndices(res))

  return State(field, fieldIterations, iterations, res, bounds, lengths, delta, deltas, op)
end

