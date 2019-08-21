function iterate!(state::State, idx)
  for I in CartesianIndices(state.res)
    res = state.op(state.field[I], state.deltas[I])
    if (abs2(res) < 4)
      state.fieldIterations[I] = idx
    end

    state.field[I] = res
  end
end

function step!(state::State)
  state.fieldIterations .= 0
  for i = 1:state.iterations
    iterate!(state, i)
  end
  state.field .= 0.0
end

