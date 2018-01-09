import Base: broadcast!, *

*(A::CuMatrix{T}, x::CuVector{T}) where T = BLAS.gemv('N', T(1), A, x)
*(A::CuMatrix{T}, B::CuMatrix{T}) where T = BLAS.gemm('N', 'N', T(1), A, B)

function broadcast!(::typeof(+), y::CuArray, x1::CuArray, x2::CuArray)
    if y === x1
        CUDNN.add!(1, x2, 1, y)
    elseif y === x2
        CUDNN.add!(1, x1, 1, y)
    else
        throw("Not implemented.")
    end
    y
end
