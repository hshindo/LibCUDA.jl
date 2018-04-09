export CuArray, CuVector, CuMatrix, CuVecOrMat
export curand, curandn

mutable struct CuArray{T,N} <: AbstractCuArray{T,N}
    ptr::CuPtr
    dims::NTuple{N,Int}
end

const CuVector{T} = CuArray{T,1}
const CuMatrix{T} = CuArray{T,2}
const CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}

function CuArray{T}(dims::NTuple{N,Int}) where {T,N}
    # ptr = alloc(getallocator(), sizeof(T)*prod(dims))
    ptr = CuPtr(sizeof(T)*prod(dims))
    CuArray{T,N}(ptr, dims)
end
CuArray{T}(dims::Int...) where T = CuArray{T}(dims)
CuArray{T}(ptr::CuPtr, dims::NTuple{N,Int}) where {T,N} = CuArray{T,N}(ptr, dims)
CuArray(x::Array{T,N}) where {T,N} = copy!(CuArray{T}(size(x)), x)
CuArray(x::CuArray) = x

Base.length(x::CuArray) = prod(x.dims)
Base.size(x::CuArray) = x.dims
function Base.size(x::CuArray{T,N}, d::Int) where {T,N}
    @assert d > 0
    d <= N ? x.dims[d] : 1
end
Base.ndims(x::CuArray{T,N}) where {T,N} = N
Base.eltype(x::CuArray{T}) where T = T
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

Base.similar(x::CuArray{T}) where T = CuArray{T}(size(x))
Base.similar(x::CuArray{T}, dims::NTuple) where T = CuArray{T}(dims)
Base.similar(x::CuArray{T}, dims::Int...) where T = similar(x, dims)

Base.convert(::Type{Ptr{T}}, x::CuArray) where T = Ptr{T}(x.ptr)
Base.unsafe_convert(::Type{Ptr{T}}, x::CuArray) where T = Ptr{T}(x.ptr)

Base.zeros(x::CuArray{T,N}) where {T,N} = zeros(CuArray{T}, x.dims)
Base.zeros(::Type{CuArray{T}}, dims::Int...) where T = zeros(CuArray{T}, dims)
Base.zeros(::Type{CuArray{T}}, dims::NTuple) where T = fill!(CuArray{T}(dims), 0)

Base.ones(x::CuArray{T}) where T = ones(CuArray{T}, x.dims)
Base.ones(::Type{CuArray{T}}, dims::Int...) where T  = ones(CuArray{T}, dims)
Base.ones(::Type{CuArray{T}}, dims::NTuple) where T = fill!(CuArray{T}(dims), 1)

function Base.copy!(dest::Array{T}, src::CuArray{T}) where T
    nbytes = length(src) * sizeof(T)
    @apicall :cuMemcpyDtoH (Ptr{Void},Ptr{Void},Csize_t) dest src nbytes # async is slower?
    dest
end
function Base.copy!(dest::CuArray{T}, src::Array{T}; stream=C_NULL) where T
    nbytes = length(src) * sizeof(T)
    @apicall :cuMemcpyHtoDAsync (Ptr{Void},Ptr{Void},Csize_t,Ptr{Void}) dest src nbytes stream
    dest
end
function Base.copy!(dest::CuArray{T}, src::CuArray{T}; stream=C_NULL) where T
    nbytes = length(src) * sizeof(T)
    @apicall :cuMemcpyDtoDAsync (Ptr{Void},Ptr{Void},Csize_t,Ptr{Void}) dest src nbytes stream
    dest
end
function Base.copy!(dest::CuArray{T}, doffs::Int, src::CuArray{T}, soffs::Int, n::Int; stream=C_NULL) where T
    p_dest = pointer(dest, doffs)
    p_src = pointer(src, soffs)
    nbytes = n * sizeof(T)
    @apicall :cuMemcpyDtoDAsync (Ptr{Void},Ptr{Void},Csize_t,Ptr{Void}) p_dest p_src nbytes stream
    dest
end
@generated function Base.copy!(dest::CuArray{T,N}, doffs::NTuple{N,Int}, src::CuArray{T,N}, soffs::NTuple{N,Int}, n::NTuple{N,Int}; stream=C_NULL) where {T,N}
    throw("Not implemented yet.")
    CT = cstring(T)
    f = compile("""
    shiftcopy(Array<$CT,3> dest, Array<$CT,3> src, Dims<3> shift) {
        int idx0 = threadIdx.x + blockIdx.x * blockDim.x;
        int idx1 = threadIdx.y + blockIdx.y * blockDim.y;
        int idx2 = threadIdx.z + blockIdx.z * blockDim.z;
        if (idx0 >= src.dims[0] || idx1 >= src.dims[1] || idx2 >= src.dims[2]) return;
        dest(idx0+shift[0], idx1+shift[1], idx2+shift[2]) = src(idx0, idx1, idx2);
    }""")
    quote
        launch($f, size(src), (dest,src,shift))
        dest
    end
end
Base.copy(src::CuArray) = copy!(similar(src), src)

Base.pointer(x::CuArray{T}, index::Int=1) where T = Ptr{T}(x) + sizeof(T)*(index-1)
Base.Array(src::CuArray{T,N}) where {T,N} = copy!(Array{T}(size(src)), src)
Base.isempty(x::CuArray) = length(x) == 0
Base.vec(x::CuArray{T}) where T = ndims(x) == 1 ? x : CuArray{T}(x.ptr, (length(x),))
Base.reshape{T,N}(a::CuArray{T}, dims::NTuple{N,Int}) = CuArray{T,N}(a.ptr, dims)
Base.reshape{T}(a::CuArray{T}, dims::Int...) = reshape(a, dims)
Base.fill(::Type{CuArray}, value::T, dims::NTuple) where T = fill!(CuArray{T}(dims), value)
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
#Base.getindex(a::CuArray, key...) = copy!(view(a, key...))

function Base.setindex!(y::CuArray{T,N}, x::CuArray{T,N}, indexes...) where {T,N}
    if N <= 3
        shift = [0,0,0]
        for i = 1:length(indexes)
            idx = indexes[i]
            if typeof(idx) == Colon
            elseif typeof(idx) <: Range
                # TODO: more range check
                @assert length(idx) == size(x,i)
                shift[i] = start(idx) - 1
            else
                throw("Invalid range: $(idx)")
            end
        end
        shiftcopy!(reshape3(y), reshape3(x), (shift[1],shift[2],shift[3]))
    else
        throw("Not implemented yet.")
    end
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
