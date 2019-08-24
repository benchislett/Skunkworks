" Coloring scheme for the plot, in RGBA "
const colorMap = [
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

"""Linearly interpolate between two colors"""
function lerp(colorA, colorB, scale)
  r = UInt8(floor((1 - scale) * colorA.r + scale * colorB.r))
  g = UInt8(floor((1 - scale) * colorA.g + scale * colorB.g))
  b = UInt8(floor((1 - scale) * colorA.b + scale * colorB.b))
  a = UInt8(floor((1 - scale) * colorA.a + scale * colorB.a))

  return sfColor(r, g, b, a)
end

"""
getColor(val::Complex{Float64}, n::Integer)

Each color is colored exponentially according to its value after the simulation, and the number of steps needed for it to exceed that max value.

The color is chosen according to a logarithmic scale and linearly interpolated with the next closest color to provide a smooth transition between colors.
"""
function getColor(val, n, nmax)
  v = abs(val)
  if v < 2
    n_log = 0
  else
    n_log = log2(log2(v))
  end

  n = n - n_log

  n = (n / nmax) * (length(colorMap) - 1)

  colorA = colorMap[Int(floor(n)) + 1]
  colorB = colorMap[Int(floor(n + 1)) % length(colorMap) + 1]

  return lerp(colorA, colorB, n % 1)
end

