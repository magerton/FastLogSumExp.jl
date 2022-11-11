const AbsFloatMat = AbstractMatrix{<:AbstractFloat}
const AbsFloatVec = AbstractVector{<:AbstractFloat}

# --------------------------------------------------
# Matrix logsumexp
# --------------------------------------------------

function mat_logsumexp_dual_reinterp!(
    Vbar::AbstractVector{D}, tmp_max::AbstractVector{V}, 
    tmpX::AbstractMatrix{V}, X::AbstractMatrix{D}, scale::Number=1
    ) where {T,V,K,D<:FD.Dual{T,V,K}}

    scale > 0 || throw(DomainError(scale, "scale must be positive"))

    m,n = size(X)

    (m,n) == size(tmpX) || throw(DimensionMismatch())
    (m,) == size(Vbar) == size(tmp_max) || throw(DimensionMismatch())

    Vre   = reinterpret(reshape, V, Vbar)
    Xre   = reinterpret(reshape, V, X)

    tmp_inv = tmp_max # resuse

    invscale = inv(scale)


    fill!(Vbar, 0)
    fill!(tmp_max, typemin(V))

    @turbo for i in 1:m, j in 1:n
        tmp_max[i] = max(tmp_max[i], Xre[1,i,j])
    end

    # set tmpX to exp((X - tmp_max)/scale)
    # and Vre[1,i] to sum(tmpX[i,:])
    @turbo for i in 1:m, j in 1:n
        ex = exp((Xre[1,i,j] - tmp_max[i])*invscale)
        tmpX[i,j] = ex
        Vre[1,i] += ex
    end

    @turbo for i in 1:m
        v = Vre[1,i]
        m = tmp_max[i]
        tmp_inv[i] = inv(v)
        Vre[1,i] = scale*log(v) + m
    end

    @turbo for i in 1:m, j in 1:n, k in 1:K
        Vre[k+1,i] += tmpX[i,j]*Xre[k+1,i,j]*tmp_inv[i]
    end

    return Vbar

end

"using base loops with LoopVectorization `exp` and `log`"
function mat_logsumexp_vexp_log_fast!(Vbar, tmp_max, X, scale::Number=1)
    scale > 0 || throw(DomainError(scale, "scale must be positive"))
    
    m,n = size(X)
    maximum!(tmp_max, X)
    fill!(Vbar, 0)

    invscale = inv(scale)

    for j in 1:n
        for i in 1:m
            Vbar[i] += vexp((X[i,j] - tmp_max[i])*invscale)
        end
    end
    for i in 1:m
        Vbar[i] = scale*log_fast(Vbar[i]) + tmp_max[i]
    end
    return Vbar
end

"""
using `LoopVectorization.@turbo` loops

**NOTE** - not compatible with `ForwardDiff.Dual` numbers!
"""
function mat_logsumexp_float_turbo!(Vbar::AbstractVector{T}, tmp_max::AbstractVector{T}, X::AbstractMatrix{T}, scale::Number=1) where {T<:AbstractFloat}

    scale > 0 || throw(DomainError(scale, "scale must be positive"))
    
    m,n = size(X)
    invscale = inv(scale)

    fill!(Vbar, 0)
    fill!(tmp_max, typemin(T))
    
    @turbo for i in 1:m, j in 1:n
        tmp_max[i] = max(tmp_max[i], X[i,j])
    end
    
    @turbo for i in 1:m, j in 1:n
        Vbar[i] += exp((X[i,j] - tmp_max[i])*invscale)
    end
    
    @turbo for i in 1:m
        Vbar[i] = scale*log(Vbar[i]) + tmp_max[i]
    end
    return Vbar
end



# --------------------------------------------------
# Matrix softmax!
# --------------------------------------------------


"faster softmax! leverages @turbo"
function mat_softmax_float_turbo!(q::AbstractMatrix{<:AbstractFloat}, tmp_max, u, scale::Number=1)
    
    scale > 0 || throw(DomainError(scale, "scale must be positive"))

    m,n = size(q)
    (m,n) == size(u) || throw(DimensionMismatch())
    length(tmp_max) == m || throw(DimensionMismatch())
    
    T = eltype(q)
    fill!(tmp_max, typemin(T))

    invscale = inv(scale)

    @turbo for i in 1:m, j in 1:n
        tmp_max[i] = max(tmp_max[i], u[i,j])
    end

    @turbo for i in 1:m, j in 1:n
        q[i,j] = exp((u[i,j] - tmp_max[i])*invscale)
    end

    fill!(tmp_max, 0)
    @turbo for i in 1:m, j in 1:n
        tmp_max[i] += q[i,j]
    end

    @turbo for i in 1:m
        s = tmp_max[i]
        invs = inv(s)
        for j in 1:n
            q[i,j] *= invs
        end
    end
    return q
end



"faster softmax! leverages @simd & @turbo where possible"
function mat_softmax_dual_reinterp!(q::AbstractMatrix{<:FD.Dual}, tmp_max, u, scale::Number=1)

    scale > 0 || throw(DomainError(scale, "scale must be positive"))

    m,n = size(q)
    (m,n) == size(u) || throw(DimensionMismatch())
    length(tmp_max) == m || throw(DimensionMismatch())
    
    tmpmaxr = reinterpret(reshape, Float64, tmp_max)
    qr = reinterpret(reshape, Float64, q)
    k = size(tmpmaxr, 1)

    T = eltype(q)
    fill!(tmp_max, typemin(T))

    invscale = inv(scale)

    @inbounds for j in 1:n
        @simd for i in 1:m
            tmp_max[i] = max(tmp_max[i], u[i,j])
        end
    end

    @inbounds for j in 1:n
        @simd for i in 1:m
            q[i,j] = exp((u[i,j] - tmp_max[i])*invscale)
        end
    end

    fill!(tmp_max, 0)
    
    @turbo for k in 1:k, i in 1:m, j in 1:n
        tmpmaxr[k,i,j] += qr[k,i,j]
    end

    @inbounds for i in 1:m
        s = tmp_max[i]
        invs = inv(s)
        @simd for j in 1:n
            q[i,j] *= invs
        end
    end
    return q
end
