module FastLogSumExp

using LoopVectorization, VectorizationBase
using LogExpFunctions, ForwardDiff, Tullio

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

"using base SIMD loops"
function logsumexp_simd!(Vbar, tmp_max, X)
	m,n, = size(X)
	maximum!(tmp_max, X)
	fill!(Vbar, 0)
	@inbounds for j in 1:n
		@simd for i in 1:m
			Vbar[i] += exp(X[i,j] - tmp_max[i])
		end
	end
	@inbounds @simd for i in 1:m
		Vbar[i] = log(Vbar[i]) + tmp_max[i]
	end
	return Vbar
end


"using base SIMD loops with LoopVectorization tricks"
function logsumexp_tricks!(Vbar, tmp_max, X)
	m,n = size(X)
	maximum!(tmp_max, X)
    # tmp_max = vreduce(max, X; dims=2)
	fill!(Vbar, 0)
	@inbounds for j in 1:n
		@simd for i in 1:m
			Vbar[i] += vexp(X[i,j] - tmp_max[i])
		end
	end
    	
	@inbounds @simd for i in 1:m
		Vbar[i] = log_fast(Vbar[i]) + tmp_max[i]
	end
	return Vbar
end

"using base SIMD loops with LoopVectorization tricks"
function logsumexp_turbo2!(Vbar, tmp_max, X)
	m,n = size(X)
	maximum!(tmp_max, X)
	fill!(Vbar, 0)
    @turbo safe=false warn_check_args=false for i in 1:m, j in 1:n
		Vbar[i] += vexp(X[i,j] - tmp_max[i])
	end
    	
	@turbo safe=false warn_check_args=false for i in 1:m
		Vbar[i] = log_fast(Vbar[i]) + tmp_max[i]
	end
	return Vbar
end


"using base SIMD loops with LoopVectorization tricks"
function logsumexp_specials!(Vbar, tmp_max, X)
    m,n = size(X)
	maximum!(tmp_max, X)
	fill!(Vbar, 0)
	for j in 1:n
		for i in 1:m
			Vbar[i] += vexp(X[i,j] - tmp_max[i])
		end
	end
	for i in 1:m
		Vbar[i] = log_fast(Vbar[i]) + tmp_max[i]
	end
	return Vbar
end


"vanilla loop with no @simd"
function logsumexp_vanilla!(Vbar, tmp_max, X)
	m,n = size(X)
	maximum!(tmp_max, X)
	fill!(Vbar, 0)
	for i in 1:m, j in 1:n
		Vbar[i] += exp(X[i,j] - tmp_max[i])
	end
	for i in 1:m
		Vbar[i] = log(Vbar[i]) + tmp_max[i]
	end
	return Vbar
end



"""
using `LoopVectorization.@turbo` loops

**NOTE** - not compatible with `ForwardDiff.Dual` numbers!
"""
function logsumexp_turbo!(Vbar, tmp_max, X)
	m,n = size(X)
	maximum!(tmp_max, X)
	fill!(Vbar, 0)
	@turbo for i in 1:m, j in 1:n
		Vbar[i] += exp(X[i,j] - tmp_max[i])
	end
	@turbo for i in 1:m
		Vbar[i] = log(Vbar[i]) + tmp_max[i]
	end
	return Vbar
end

 
"""
using `LoopVectorization` `vmap` convenience fcts

**NOTE** - this DOES work with `ForwardDiff.Dual` numbers!
"""
function logsumexp_vmap!(Vbar, tmp_max, X, Xtmp)
	maximum!(tmp_max, X)
	n = size(X,2)
	for j in 1:n
		Xtmpj = view(Xtmp, :, j)
		Xj    = view(X, :, j)
		vmap!((xij, mi) -> exp(xij-mi), Xtmpj, Xj, tmp_max)
	end
	Vbartmp = vreduce(+, Xtmp; dims=2)
	vmap!((vi,mi) -> log(vi) + mi, Vbar, Vbartmp, tmp_max)
	return Vbar
end

"Using tullio"
function logsumexp_tullio!(Vbar, tmp_max, X)
    @tullio avx=true (max) tmp_max[i] = X[i,j]
    @tullio avx=true Vbar[i] = exp(X[i,j] - tmp_max[i])
	@tullio avx=true Vbar[i] = log1p(Vbar[i]-1) + tmp_max[i]
  end



end # module FastLogSumExp
