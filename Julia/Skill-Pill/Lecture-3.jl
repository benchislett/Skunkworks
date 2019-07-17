# Monte Carlo integration of x^2 on the range [-diameter/2, diameter/2)
function monte_carlo(diameter::Float64, iterations::Int64)
  # Precompute radius and squared radius
  radius::Float64 = diameter / 2.0
  rad2::Float64 = radius * radius
  count::Int64 = 0

  # Iterate through a number of random trials
  for i = 1:iterations
    # Exploit the symmetry of x^2
    x::Float64 = rand() * radius
    y::Float64 = rand() * rad2

    # If a random point is below the curve, increment
    if (y < x * x)
       count += 1
    end
  end

  # A_box = r^2
  # Ratio = (A_curve / A_box) / r
  # A_curve = Ratio * A_box * r
  # A_curve = Ratio * r^3
  # Doubling since we only simulate on the positive half
  return 2 * (count / iterations) * rad2 * radius
end

# using BenchmarkTools
# println(example_monte_carlo(6.0, 100000))
# @benchmark example_monte_carlo(6.0, 100000)


using Images
# Julia set simulation on [-2, 2]
function julia(iterations::Int64, resolution::Int64, c::ComplexF64)
  # Allocate z only once
  z::ComplexF64 = 0.0

  # Create output array, only on range [0,1] since that is what Images requires
  out = ones(Float64, 3, resolution, resolution)
  # Iterate over the image space
  for i=1:resolution
    for j=1:resolution
      n::Int64 = 1
      # Set z to the coordinates with real part j and imaginary part i, bounded from -2 to 2
      z = ComplexF64(4.0 * (j - 1 - 0.5 * resolution) / resolution) + ComplexF64(4.0 * (i - 1 - 0.5 * resolution) / resolution) * 1.0im
      while n < iterations
        # Apply iteration to z
        z = z^2 + c
        
        # Exit early if z gets too big
        if (abs(z) > 2)
          break
        end
        n += 1
      end
      # Apply the number of iterations taken as a fraction from [0,1] to the value of the pixel
      out[3, i, j] = Float64(n) / Float64(iterations)
    end
  end

  # Cast to HSV and save file
  save("./output/julia.png", colorview(HSV, out))
end

# julia(100, 2048, 0.0 + 0.75im)
