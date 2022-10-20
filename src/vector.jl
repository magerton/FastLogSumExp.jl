# --------------------------------------------------
# Vector logsumexp
# --------------------------------------------------

"fastest logsumexp over Dual vector requires tmp vector"
function vec_logsumexp_dual_reinterp!(tmp::AbstractVector{V}, X::AbstractVector{<:FD.Dual{T,V,K}}) where {T,V,K}
    Xre   = reinterpret(reshape, V, X)

    uv = typemin(V)
    @turbo for i in eachindex(X)
        uv = max(uv, Xre[1,i])
    end

    s = zero(V)

    @turbo for j in eachindex(X,tmp)
        ex = exp(Xre[1,j] - uv)
        tmp[j] = ex
        s += ex
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
function vec_logsumexp_dual_reinterp(X::AbstractVector{<:FD.Dual{T,V,K}}) where {T,V,K}
    tmp = Vector{V}(undef, length(X))
    return vec_logsumexp_dual_reinterp!(tmp, X)
end

"logsumexp with @turbo. maybe a bit less safe/stable... but REALLY fast!"
function vec_logsumexp_float_turbo(x::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)
    u = maximum(x)                                       # max value used to re-center
    
    s = zero(T)
    @turbo for i = 1:n
        tmp = exp(x[i] - u)
        s += tmp
    end

    return log(s) + u
end
