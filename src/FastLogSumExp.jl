module FastLogSumExp

using LoopVectorization, VectorizationBase
using LogExpFunctions, ForwardDiff, Tullio, StaticArrays

const FD = ForwardDiff

import VectorizationBase: vexp
import SLEEFPirates: log_fast

@inline function vexp(d::FD.Dual{T}) where {T}
    val = vexp(FD.value(d))
    partials =  FD.partials(d)
    return FD.Dual{T}(val, val * partials)
end

@inline function log_fast(d::FD.Dual{T}) where {T}
    val = FD.value(d)
    partials =  FD.partials(d)
    return FD.Dual{T}(log_fast(val), inv(val) * partials)
end

include("vector.jl")
include("matrix.jl")

end # module FastLogSumExp
