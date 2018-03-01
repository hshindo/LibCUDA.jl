export CuSubArray, CuSubVector, CuSubMatrix, CuSubVecOrMat

type CuSubArray{T,N} <: AbstractCuArray{T,N}
    parent::CuArray{T}
    indexes::Tuple
    offset::Int
    dims::NTuple{N,Int}
    strides::NTuple{N,Int}
end

const CuSubVector{T} = CuSubArray{T,1}
const CuSubMatrix{T} = CuSubArray{T,2}
const CuSubVecOrMat{T} = Union{CuSubVector{T},CuSubMatrix{T}}

Base.size(a::CuSubArray) = a.dims
Base.size(a::CuSubArray, dim::Int) = a.dims[dim]
Base.strides(a::CuSubArray) = a.strides
Base.strides(a::CuSubArray, dim::Int) = a.strides[dim]
Base.length(a::CuSubArray) = prod(a.dims)
Base.similar(a::CuSubArray) = similar(a, size(a))
Base.similar(a::CuSubArray, dims::Int...) = similar(a, dims)
Base.similar{T,N}(a::CuSubArray{T}, dims::NTuple{N,Int}) = CuArray(T, dims)

Base.convert(::Type{Ptr{T}}, x::CuSubArray{T}) where T = pointer(x.parent, x.offset+1)
Base.convert(::Type{UInt64}, x::CuSubArray) = UInt64(pointer(x.parent,x.offset+1))
Base.unsafe_convert(::Type{Ptr{T}}, x::CuSubArray) where T = pointer(x.parent, x.offset+1)
Base.unsafe_convert(::Type{UInt64}, x::CuSubArray) = UInt64(pointer(x.parent,x.offset+1))

#=
function Base.pointer(a::CuSubArray, index::Int=1)
    index == 1 && return pointer(a.parent, a.offset+index)
    throw("Not implemented yet.")
end

function Base.pointer(a::CuSubArray, index::Int=1)
    index == 1 && return cupointer(a.parent, a.offset+index)
    throw("Not implemented yet.")
end
=#

function Base.view(x::CuArray{T,N}, indexes...) where {T,N}
    dims = Int[]
    strides = Int[]
    stride = 1
    offset = 0
    for i = 1:length(indexes)
        r = indexes[i]
        if isa(r, Colon)
            push!(dims, size(x,i))
            push!(strides, stride)
        elseif isa(r, Int)
            offset += stride * (r-1)
        elseif isa(r, UnitRange{Int})
            push!(dims, length(r))
            push!(strides, stride)
            offset += stride * (first(r)-1)
        else
            throw("Invalid index: $t.")
        end
        stride *= size(x,i)
    end
    CuSubArray(x, indexes, offset, tuple(dims...), tuple(strides...))
end

Base.Array(a::CuSubArray) = Array(a.parent)[a.indexes...]

function CuArray(x::CuSubArray{T}) where T
    y = CuArray{T}(size(x))
    copy!(y, x)
    y
end

Base.show(io::IO, ::Type{CuSubArray{T,N}}) where {T,N} = print(io, "CuSubArray{$T,$N}")
function Base.showarray(io::IO, X::CuSubArray, repr::Bool=true; header=true)
    if repr
        print(io, "CuSubArray(")
        Base.showarray(io, Array(X), true)
        print(io, ")")
    else
        header && println(io, summary(X), ":")
        Base.showarray(io, Array(X), false, header = false)
    end
end
