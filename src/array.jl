export CuArray, CuVector, CuMatrix, CuVecOrMat
export curand, curandn

mutable struct CuArray{T,N} <: AbstractCuArray{T,N}
    ptr::CuPtr{T}
    dims::NTuple{N,Int}
end

const CuVector{T} = CuArray{T,1}
const CuMatrix{T} = CuArray{T,2}
const CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}

function CuArray{T}(dims::NTuple{N,Int}) where {T,N}
    # ptr = alloc(getallocator(), sizeof(T)*prod(dims))
    ptr = CuPtr{T}(prod(dims))
    #strides = Array{Int}(N)
    #strides[1] = 1
    #for i = 2:length(strides)
    #     strides[i] = strides[i-1] * dims[i-1]
    #end
    CuArray(ptr, dims)
end
CuArray{T}(dims::Int...) where T = CuArray{T}(dims)
CuArray(x::Array{T,N}) where {T,N} = copy!(CuArray{T}(size(x)), x)
CuArray(x::CuArray) = x

function Base.stride(x::CuArray, dim::Int)
    d = 1
    for i = 1:dim-1
        d *= size(x, i)
    end
    d
end
Base.strides(x::CuArray{T,1}) where T = (1,)
Base.strides(x::CuArray{T,2}) where T = (1,size(x,1))
Base.strides(x::CuArray{T,3}) where T = (1,size(x,1),size(x,1)*size(x,2))
Base.strides(x::CuArray{T,4}) where T = (1,size(x,1),size(x,1)*size(x,2),size(x,1)*size(x,2)*size(x,3))

Base.convert(::Type{Ptr{T}}, x::CuArray) where T = Ptr{T}(pointer(x))
Base.unsafe_convert(::Type{Ptr{T}}, x::CuArray) where T = Ptr{T}(pointer(x))

Base.pointer(x::CuArray{T}, index::Int=1) where T = x.ptr.ptr + sizeof(T)*(index-1)
Base.vec(x::CuArray{T}) where T = ndims(x) == 1 ? x : CuArray(x.ptr, (length(x),))
Base.reshape{T,N}(a::CuArray{T}, dims::NTuple{N,Int}) = CuArray{T,N}(a.ptr, dims)
Base.reshape{T}(a::CuArray{T}, dims::Int...) = reshape(a, dims)
Base.fill(::Type{CuArray}, value::T, dims::NTuple) where T = fill!(CuArray{T}(dims), value)

Base.zeros(::Type{CuArray{T}}, dims::Int...) where T = zeros(CuArray{T}, dims)
Base.zeros(::Type{CuArray{T}}, dims::NTuple) where T = fill!(CuArray{T}(dims), 0)
Base.ones(::Type{CuArray{T}}, dims::Int...) where T  = ones(CuArray{T}, dims)
Base.ones(::Type{CuArray{T}}, dims::NTuple) where T = fill!(CuArray{T}(dims), 1)

#=
function Base.fill!(x::CuArray{T}, value; stream=C_NULL) where T
    s = sizeof(T)
    if s == 4
        @apicall :cuMemsetD32Async (Ptr{Void},Cuint,Csize_t,Ptr{Void}) x value length(x) stream
    elseif s == 2
        @apicall :cuMemsetD16Async (Ptr{Void},Cushort,Csize_t,Ptr{Void}) x value length(x) stream
    elseif s == 1
        @apicall :cuMemsetD8Async (Ptr{Void},Cuchar,Csize_t,Ptr{Void}) x value length(x) stream
    end
    x
end
=#

function Base.getindex(x::CuArray, indexes...)
    src = view(x, indexes...)
    dest = similar(x, size(src))
    copy!(dest, src)
end
function Base.getindex(x::CuArray{T}, index::Int) where T
    dest = copy!(Array{T}(1), x, 1)
    dest[1]
end

function Base.setindex!(dest::CuArray{T}, src::CuArray{T}, indexes...) where T
    copy!(view(dest,indexes...), src)
end

Base.show(io::IO, ::Type{CuArray{T,N}}) where {T,N} = print(io, "CuArray{$T,$N}")
function Base.showarray(io::IO, X::CuArray, repr::Bool=true; header=true)
    if repr
        print(io, "CuArray(")
        Base.showarray(io, Array(X), true)
        print(io, ")")
    else
        header && println(io, summary(X), ":")
        Base.showarray(io, Array(X), false, header = false)
    end
end

function curand(::Type{T}, dims::NTuple{N,Int}) where {T,N}
    # TODO: use curand library
    CuArray(rand(T,dims))
end
curand(::Type{T}, dims::Int...) where T = curand(T, dims)
curand(dims::Int...) = curand(Float64, dims)

function curandn(::Type{T}, dims::NTuple{N,Int}) where {T,N}
    # TODO: use curand library
    CuArray(randn(T,dims))
end
curandn(::Type{T}, dims::Int...) where T = curandn(T, dims)
curandn(dims::Int...) = curandn(Float64, dims)
