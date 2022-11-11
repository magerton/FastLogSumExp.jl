# --------------------------------------------------
# Vector logsumexp
# --------------------------------------------------

"fastest logsumexp over Dual vector requires tmp vector"
function vec_logsumexp_dual_reinterp!(tmp::AbstractVector{V}, X::AbstractVector{<:FD.Dual{T,V,K}}, scale::Number=1) where {T,V,K}
    scale > 0 || throw(DomainError(scale, "scale must be positive"))
    
    Xre   = reinterpret(reshape, V, X)

    invscale = inv(scale)

    uv = typemin(V)
    @turbo for i in eachindex(X)
        uv = max(uv, Xre[1,i])
    end
    uvscaled = uv*invscale

    s = zero(V)

    @turbo for j in eachindex(X,tmp)
        ex = exp(Xre[1,j]*invscale - uvscaled)
        tmp[j] = ex
        s += ex
    end

    v = scale*log(s) + uv # logsumexp value

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
function vec_logsumexp_dual_reinterp(X::AbstractVector{<:FD.Dual{T,V,K}}, scale::Number=1) where {T,V,K}
    tmp = Vector{V}(undef, length(X))
    return vec_logsumexp_dual_reinterp!(tmp, X, scale)
end

"logsumexp with @turbo. maybe a bit less safe/stable... but REALLY fast!"
function vec_logsumexp_float_turbo(x::AbstractVector{T}, scale::Number=1) where {T<:AbstractFloat}
    scale > 0 || throw(DomainError(scale, "scale must be positive"))
    
    n = length(x)
    u = maximum(x)                                       # max value used to re-center
    
    invscale = inv(scale)
    uscaled = u*invscale

    s = zero(T)
    @turbo for i = 1:n
        tmp = exp(x[i]*invscale - uscaled)
        s += tmp
    end

    return scale*log(s) + u
end



"softmax with @turbo. maybe a bit less safe/stable... but REALLY fast!"
function vec_softmax_float_turbo!(r::AbstractVector, x::AbstractVector{T}, scale::Number=1) where {T<:AbstractFloat}
    scale > 0 || throw(DomainError(scale, "scale must be positive"))

    n = length(x)
    u = maximum(x) # max value used to re-center

    invscale = inv(scale)
    uscaled = u*invscale

    s = zero(T)
    
    @turbo for i = 1:n
        tmp = exp(x[i]*invscale - uscaled)
        r[i] = tmp
        s += tmp
    end

    invs = inv(s) # for doing softmax for derivatives

    @turbo for i in 1:n
        r[i] *= invs
    end

    return r
end

vec_softmax_float_turbo(x::AbstractVector{<:AbstractFloat}, scale::Number=1) = 
    vec_softmax_float_turbo!(similar(x), x, scale)