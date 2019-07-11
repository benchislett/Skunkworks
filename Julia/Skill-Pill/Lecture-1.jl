# First session, 10am-12pm, Thursday July 11, 2019

x = 1 # Variable declaration, type inference to Int64
y = 1.0 # type inference to Float64
z = 1.0 - 2.0im # type inference to Complex{Float64}

a = 1//2 # Fraction literal
b = "abc" # String literal
c = 1.0f0 # Float32 literal

"""
multiline comments
"""

typeof(x) # Type operator

# LaTeX symbol definition: backslash followed by character name, then press tab
ϵ = 1e-12
ϕ = 16.0
println(ϕ, " - ", ϵ, " = ", ϕ - ϵ)
# NOTE: Don't overuse symbols for the sake of readability and those whose editors don't support latex chars


# Conditional statements
if x < 2 # No colon or braces
  x = 3
else
  x = 1
end # Must have an "end"

# Print to stdout
println("Testing print")

# For loops
for (i, x) in enumerate(['A', 'B', 'C']) # Array literal
  if x == 'B'
    println("Found b at position: ", i)
  end
end # loops must end also

while true
  # Ternary operator
  rand() < 0.1 ? break : continue
end

# Functions
function add(x, y)
  return x + y
end

# Easy function definitions
g(x) = x^2
doubled = map((x)->2*x,[1,2,3])

# map(function, array): apply function to each element in array and return new array
lowers = map(lowercase, ['A', 'B', 'C'])

# Abstract types: groups of types
abstract type Entity end
mutable struct Player <: Entity
  mass::Float64
  name::String
end
struct Object <: Entity # immutable struct: entries are readonly after construction
  mass::Float64
end

# Typed function parameters
function(x::Number)
  println("x is a number")
end

# modules: namespaces
module myModule
  export f

  function f(x::Number)
    println(x)
  end
end

# In REPL: type "?" to get into help prompt, then enter method names for help
# In REPL: type "]" to get into pkg prompt
# In REPL: type ";" to get into shell prompt

# Load packages (must be installed with pkg if not available by default
using Plots

# TODO: find how to define help statement for custom modules

res = 3 \ 7 # Inverse division (3 \ 7 == 7 / 3)
