const AbsFloatMat = AbstractMatrix{<:AbstractFloat}
const AbsFloatVec = AbstractVector{<:AbstractFloat}

# --------------------------------------------------
# Matrix logsumexp
# --------------------------------------------------

function mat_logsumexp_dual_reinterp!(
    Vbar::AbstractVector{D}, tmp_max::AbstractVector{V}, 
    tmpX::AbstractMatrix{V}, X::AbstractMatrix{D}
    ) where {T,V,K,D<:FD.Dual{T,V,K}}
    
    m,n = size(X)

    (m,n) == size(tmpX) || throw(DimensionMismatch())
    (m,) == size(Vbar) == size(tmp_max) || throw(DimensionMismatch())

    Vre   = reinterpret(reshape, V, Vbar)
    Xre   = reinterpret(reshape, V, X)

    tmp_inv = tmp_max # resuse

    fill!(Vbar, 0)
    fill!(tmp_max, typemin(V))

    @turbo for i in 1:m, j in 1:n
        tmp_max[i] = max(tmp_max[i], Xre[1,i,j])
    end

    @turbo for i in 1:m, j in 1:n
        ex = exp(Xre[1,i,j] - tmp_max[i])
        tmpX[i,j] = ex
        Vre[1,i] += ex
    end

    @turbo for i in 1:m
        v = Vre[1,i]
        m = tmp_max[i]
        tmp_inv[i] = inv(v)
        Vre[1,i] = log(v) + m
    end

    @turbo for i in 1:m, j in 1:n, k in 1:K
        Vre[k+1,i] += tmpX[i,j]*Xre[k+1,i,j]*tmp_inv[i]
    end

    return Vbar

end

"using base loops with LoopVectorization `exp` and `log`"
function mat_logsumexp_vexp_log_fast!(Vbar, tmp_max, X)
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

"""
using `LoopVectorization.@turbo` loops

**NOTE** - not compatible with `ForwardDiff.Dual` numbers!
"""
function mat_logsumexp_float_turbo!(Vbar::AbstractVector{T}, tmp_max::AbstractVector{T}, X::AbstractMatrix{T}) where {T<:AbstractFloat}
    m,n = size(X)

    fill!(Vbar, 0)
    fill!(tmp_max, typemin(T))
    
    @turbo for i in 1:m, j in 1:n
        tmp_max[i] = max(tmp_max[i], X[i,j])
    end
    
    @turbo for i in 1:m, j in 1:n
        Vbar[i] += exp(X[i,j] - tmp_max[i])
    end
    
    @turbo for i in 1:m
        Vbar[i] = log(Vbar[i]) + tmp_max[i]
    end
    return Vbar
end

