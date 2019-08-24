mutable struct State
  field::Array{Complex{Float64}, 2}
  fieldIterations::Array{Int64, 2}

  iterations::Int64

  res::Tuple{Int64, Int64}

  bounds::Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}
  lengths::Tuple{Float64, Float64}
  
  delta::Tuple{Float64, Float64}
  deltas::Array{Complex{Float64}, 2}

  op::Function
end

""" Propagate changes in the core state (res, bounds) to dependents"""
function updateState!(state::State)
  state.field .= 0
  state.fieldIterations .= 0

  state.lengths = map(b -> b[2] - b[1], state.bounds)
  
  state.delta = state.lengths ./ state.res
  for i=1:state.res[1]
    for j=1:state.res[2]
      state.deltas[i, j] = complex(state.bounds[1][1] + i * state.delta[1], state.bounds[2][1] + j * state.delta[2])
    end
  end
end

function State(; iterations=20.0, res=(32, 32), bounds=((-2, 2), (-2, 2)), op=(z,c)->z^2+c)
  field = Array{Complex{Float64}}(undef, res)
  fieldIterations = Array{Int64}(undef, res)
  
  deltas = Array{Complex{Float64}}(undef, res)

  state = State(field, fieldIterations, iterations, res, bounds, (-1.0, -1.0), (-1.0, -1.0), deltas, op)

  updateState!(state)

  return state
end

