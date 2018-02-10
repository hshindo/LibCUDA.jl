import Base.LinAlg.BLAS: axpy!

#=
for (fname,elty) in (
    (:cublasDcopy,:Float64),
    (:cublasScopy,:Float32))
    @eval begin
        function blascopy!(n::Int, x::CuArray{$elty}, incx::Int,
            y::CuArray{$elty}, incy::Int) where $elty
            $fname(handle(x), n, x, incx, y, incy)
            y
        end
    end
end
=#

for (f,T,Ct) in (
    (:(:cublasDaxpy),:Float64,:Cdouble),
    (:(:cublasSaxpy),:Float32,:Cfloat))
    @eval begin
        function axpy!(n::Int, alpha::$T, x::CuArray{$T}, incx::Int,
            y::CuArray{$T}, incy::Int)
            @cublas($f, (
                Ptr{Void},Cint,Ptr{$Ct},Ptr{$Ct},Cint,Ptr{$Ct},Cint),
                gethandle(), n, [alpha], x, incx, y, incy)
            y
        end
    end
end

function axpy!(alpha::T, x::CuArray{T}, y::CuArray{T}) where T
    length(x) == length(y) || throw(DimensionMismatch())
    axpy!(length(x), alpha, x, 1, y, 1)
end

function axpy!(alpha::T, x::CuArray{T}, rx::Range{Int}, y::CuArray{T}, ry::Range{Int}) where T
    length(rx) == length(ry) || throw(DimensionMismatch())
    (minimum(rx) < 1 || maximum(rx) > length(x)) && throw(BoundsError())
    (minimum(ry) < 1 || maximum(ry) > length(y)) && throw(BoundsError())
    axpy!(length(rx), alpha, pointer(x,first(rx)-1), step(rx), pointer(y,first(ry)-1), step(ry))
end
