import Base: broadcast!, +, -, *, /

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

function +(x1::CuArray{T,N}, x2::CuArray{T,N}) where {T,N}
    @assert length(x1) == length(x2)
    y = zeros(x1)
    BLAS.axpy!(T(1), x1, y)
    BLAS.axpy!(T(1), x2, y)
    y
end

function -(x1::CuArray{T,N}, x2::CuArray{T,N}) where {T,N}
    @assert length(x1) == length(x2)
    y = zeros(x1)
    BLAS.axpy!(T(1), x1, y)
    BLAS.axpy!(T(-1), x2, y)
    y
end

function -(x::CuArray{T}) where T
    y = zeros(x)
    BLAS.axpy!(T(-1), x, y)
    y
end
