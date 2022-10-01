# --------------------------------------------------
# Vector logsumexp
# --------------------------------------------------

"fastest logsumexp over Dual vector requires tmp vector"
function logsumexp_reinterp2!(tmp::AbstractVector{V}, X::AbstractVector{<:FD.Dual{T,V,K}}) where {T,V,K}
	Xre   = reinterpret(reshape, V, X)

	u = maximum(X)
	uv = FD.value(u)

	s = zero(V)
	@turbo for j in eachindex(X,tmp)
		ex = exp(Xre[1,j] - uv)
		tmp[j] = ex
		s  += ex
	end

	v = log(s) + uv # logsumexp value

	invs = inv(s) # for doing softmax for derivatives

	# would be nice to use a more elegant consruction for
	# pvec instead of multiple conversions below
	# that said, it seems like we still have zero allocations
	pvec = zeros(MVector{K,V})
	@turbo for j in eachindex(X,tmp)
		tmp[j] *= invs
		for k in 1:K
			pvec[k] += tmp[j]*Xre[k+1,j]
		end
	end

	ptup = NTuple{K,V}(pvec)
	ptl = FD.Partials{K,V}(ptup)

	return FD.Dual{T,V,K}(v, ptl)

end

"wrapper allocates tmp vector"
function logsumexp_reinterp2!(X::AbstractVector{<:FD.Dual{T,V,K}}) where {T,V,K}
	tmp = Vector{V}(undef, length(X))
	return logsumexp_reinterp2!(tmp, X)
end

"logsumexp with @turbo. maybe a bit less safe/stable... but REALLY fast!"
function turbologsumexp(x::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)
    u = maximum(x)                                       # max value used to re-center
    
	s = zero(T)
    @turbo for i = 1:n
        tmp = exp(x[i] - u)
        s += tmp
    end

    return log(s) + u
end

"slower logsumexp doesn't allocate, but not sure how to handle partials elegantly"
function logsumexp_reinterp1!(X::AbstractVector{<:FD.Dual{T,V,K}}) where {T,V,K}
	Xre   = reinterpret(reshape, V, X)
	n = length(X)
	@assert (K+1, n) == size(Xre)

	u = maximum(X)
	uv = FD.value(u)

	s = zero(V)
	p = zeros(MVector{K,V})
	for j in eachindex(X)
		tmp = exp(Xre[1,j] - uv)
		s  += tmp
		@turbo for k in 1:K
			p[k] += tmp * Xre[k+1,j]
		end
	end

	invs = inv(s)
	@turbo p .*= invs	
	v = log(s[1]) + uv

	ptup = NTuple{K,V}(p)
	ptl = FD.Partials{K,V}(ptup)

	return FD.Dual{T,V,K}(v, ptl)

end

"middle case for logsumexp. Mutates X in place instead of using tmp vector"
function logsumexp_reinterp3!(X::AbstractVector{<:FD.Dual{T,V,K}}) where {T,V,K}
	Xre   = reinterpret(reshape, V, X)
	n = length(X)
	@assert (K+1, n) == size(Xre)

	u = maximum(X)
	uv = FD.value(u)

	s = zero(V)
	@turbo for j in 1:n
		tmp = exp(Xre[1,j] - uv)
		Xre[1,j] = tmp
		s  += tmp
	end

	invs = inv(s)
	v = log(s) + uv # logsumexp

	p = zeros(MVector{K,V})
	@turbo for j in eachindex(X)
		Xre[1,j] *= invs
		for k in eachindex(p)
			p[k] += Xre[k+1,j]*Xre[1,j]
		end
	end

	ptup = NTuple{K,V}(p)
	ptl = FD.Partials{K,V}(ptup)

	return FD.Dual{T,V,K}(v, ptl)

end