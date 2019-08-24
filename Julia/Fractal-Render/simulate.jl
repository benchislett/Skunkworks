function iterate!(state::State, idx)
  for I in CartesianIndices(state.res)
    val = state.op(state.field[I], state.deltas[I])
    if (abs2(val) <= 65536)
      state.fieldIterations[I] = idx
      state.field[I] = val
    end
  end
end

function step!(state::State)
  for i = 1:state.iterations
    iterate!(state, i)
  end
end

