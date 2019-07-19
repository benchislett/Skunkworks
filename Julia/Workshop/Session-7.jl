# Multithreading in Julia (OIST) Friday July 19, 2019
# by Valentin Churavy, JuliaLab, CSAIL, MIT

# Julia ^1.2 recommended

NCPU = Sys.CPU_THREADS
using Base.Threads
@show NCPU
@show nthreads()

# Set environment variable `JULIA_NUM_THREADS=N` for best effects

# Fork-join parallelism
# Describes the control flow of a group of threads
# Execution is forked and an anonymous function is then run across all threads
# All threads have to join together and serial execution continues

# Special care must be taken if the loop has side-effects or accesses global state
# (i.e. IO, RNG)

function f()
  a = zeros(Int, nthreads()*3)
  @threads for i=1:length(a)
    a[i] = threadid()
  end
  return a
end

@show f()

function ff()
  acc = 0
  @threads for i=1:10_000
    acc += 1
  end
  return acc
end

@show ff()

# Atomics

function fff()
  acc = Ref(0)
  @threads for i=1:10_000
    acc[] += 1
  end
  return acc
end

@show fff()

function ffff()
  acc = Atomic{Int64}(0)
  @threads for i=1:10_000
    atomic_add!(acc, 1)
  end
  return acc
end

@show ffff()

# Locks

struct Acc{T, L}
  x::Base.RefValue{T}
  lock::L
end

Base.lock(a::Acc) = lock(a.lock)
Base.unlock(a::Acc) = unlock(a.lock)

function g(acc)
  @threads for i=1:10_000
    lock(acc)
    acc.x[] += 1
    unlock(acc)
  end
  return acc
end

@show g(Acc(Ref(0), Mutex())).x[]

@show g(Acc(Ref(0), SpinLock())).x[]

# Custom thread splitting
#=
@threads for i=1:nthreads()
  ...
end
=#

ch = Channel{Int64}(1)
done = false
@sync begin
  @async begin
    while isopen(ch)
      @show take!(ch)
    end
  end

  @async begin
    for i=1:4
      put!(ch, i)
    end
    close(ch)
  end
end

;

