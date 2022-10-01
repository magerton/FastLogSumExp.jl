# --------------------------------------------------
# Matrix logsumexp
# --------------------------------------------------

function logsumexp_reinterp!(Vbar::AbstractVector{D}, tmp_max::AbstractVector{D}, X::AbstractMatrix{D}) where {T,V,N,D<:FD.Dual{T,V,N}}
	Vre   = reinterpret(reshape, V, Vbar)
	tmpre = reinterpret(reshape, V, tmp_max)
	Xre   = reinterpret(reshape, V, X)

	m,n = size(X)
	@assert (N+1, m, n) == size(Xre)

	maximum!(tmp_max, X)
	fill!(Vbar, 0)

	@turbo for i in 1:m, j in 1:n
		Vre[1,i] += exp(Xre[1,i,j] - tmpre[1,i])
	end

	@turbo for i in 1:m
		Vbar[1,i] = log(Vbar[i]) + tmp_max[i]
	end

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
