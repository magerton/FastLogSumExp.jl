This package includes specialized functions to handle `logsumexp(X::AbstractVector})` and `logsumexp(X::AbstractMatrix; dims=2)` for both `Float64` and `ForwardDiff.Dual` numbers. These versions are 5-10x faster than `LogExpFunctions.logsumexp`. Uses `LoopVectorization.@turbo`, as well as (in the background) `VectorizationBase.vexp` and `SLEEFPirates.log_fast`. 

See issue at <https://github.com/JuliaSIMD/LoopVectorization.jl/issues/437>. Thanks, @chriselrod for pointing me to <https://github.com/PumasAI/SimpleChains.jl/blob/main/src/forwarddiff_matmul.jl>.

Benchmarks:

```julia
  "M" => 2-element BenchmarkTools.BenchmarkGroup:
          tags: ["M", "Matrix"]
          "Float64" => 3-element BenchmarkTools.BenchmarkGroup:
                  tags: ["Float64"]
                  "LogExpFunctions" => Trial(28.400 μs)
                  "Fast LogExp" => Trial(11.300 μs)
                  "Turbo" => Trial(5.100 μs)
          "Dual" => 3-element BenchmarkTools.BenchmarkGroup:
                  tags: ["Dual"]
                  "Reinterp" => Trial(12.100 μs)
                  "LogExpFunctions" => Trial(56.100 μs)
                  "Fast LogExp" => Trial(26.400 μs)
  "V" => 2-element BenchmarkTools.BenchmarkGroup:
          tags: ["V", "Vector"]
          "Float64" => 2-element BenchmarkTools.BenchmarkGroup:
                  tags: ["Float64"]
                  "LogExpFunctions" => Trial(5.900 μs)
                  "Turbo" => Trial(1.700 μs)
          "Dual" => 3-element BenchmarkTools.BenchmarkGroup:
                  tags: ["Dual"]
                  "Reinterp" => Trial(2.300 μs)
                  "LogExpFunctions" => Trial(11.800 μs)
                  "Reinterp no tmp" => Trial(2.300 μs)
```