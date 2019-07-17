# Performance Engineering in Julia (OIST) 1pm-3pm Wednesday July 17, 2019
# by Valentin Churavy, JuliaLab, CSAIL, MIT

# GEMM (General Matrix Multiplication) in Julia
# Note: ! is a convention to indicate a mutation of a parameter
function mygemm!(C, A, B)
  for j in 1:size(C, 2)
    for k in 1:size(A, 2)
      for i in 1:size(C, 1)
        # @inbounds macro: ignore boundary checks
        @inbounds C[i, j] += A[i, j] * B[k, j]
      end
    end
  end
  return C
end

# @time mygemm!(C, A, B)
# @time A*B # Using OpenBLAS

# Is Julia fast?
# yes.

# Profiler: https://docs.julialang.org/en/latest/manual/profile
# BenchmarkTools.jl
# @benchmark macro: repeats an operation to find accurate time

# Macros: interact with the AST

# @code_lowered: View the lowered AST of the code
# @code_typed optimize=bool: See the typed lowered AST, and optimize by default
# @code_warntype: See the typed lowered AST with optimization warnings on type unions
# @code_llvm optimize=bool: See the LLVM (Lower Level Virtual Machine) code
# @code_native See the native instructions

# Performance enhancing info: https://docs.julialang.org/en/v1/manual/performance-tips


