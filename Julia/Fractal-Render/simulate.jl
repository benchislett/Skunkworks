function iterate!(state::State, idx)
  for I in CartesianIndices(state.res)
    res = state.op(state.field[I], state.deltas[I])
    if (abs2(res) < 4)
      state.fieldIterations[I] = idx
    end
  end
end

function step!(state::State)
  for i = 1:state.iterations
    iterate!(state, i)
  end
end

