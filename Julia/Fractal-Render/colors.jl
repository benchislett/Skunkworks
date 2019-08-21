" Coloring scheme for the plot, in RGBA "
colorMap = [
  sfColor(0,0,0,255),
  sfColor(66,46,15,255),
  sfColor(26,8,26,255),
  sfColor(10,0,46,255),
  sfColor(5,5,74,255),
  sfColor(0,8,99,255),
  sfColor(13,43,138,255),
  sfColor(56,125,209,255),
  sfColor(133,181,230,255),
  sfColor(209,235,247,255),
  sfColor(240,232,191,255),
  sfColor(247,201,94,255),
  sfColor(255,171,0,255),
  sfColor(204,128,0,255),
  sfColor(153,87,0,255),
  sfColor(105,51,3,255)
];

"""
  getColor(state::State, n::Integer)

Each pixel is colored according to how many iterations were needed to exceed the threshold.
The more steps it took, the more extreme the color.

The color is chosen linearly according to the above mapping, with the bounds being 1 and the state's maximum for iterations

"""
function getColor(state, n)
  idx = floor(length(colorMap) * n / (state.iterations + 1))
  return colorMap[idx]
end


