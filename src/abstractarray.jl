abstract type AbstractCuArray{T,N} <: AbstractArray{T,N} end

const AbstractCuVector{T} = AbstractCuArray{T,1}
const AbstractCuMatrix{T} = AbstractCuArray{T,2}
const AbstractCuVecOrMat{T} = Union{AbstractCuVector{T},AbstractCuMatrix{T}}

@generated function Base.fill!(x::AbstractCuArray{T,N}, value) where {T,N}
    Ct = cstring(T)
    f = CuFunction("""
    $Array_h
    __global__ void f(Array<$Ct,$N> x, $Ct value) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= x.length()) return;

        x[idx] = value;
    }""")
    quote
        gdims, bdims = cudims(length(x))
        culaunch($f, gdims, bdims, x, T(value))
        x
    end
end

@generated function Base.copy!(y::AbstractCuArray{T,N}, x::AbstractCuArray{T,N}) where {T,N}
    Ct = cstring(T)
    f = CuFunction("""
    $Array_h
    __global__ void copy(Array<$Ct,$N> y, Array<$Ct,$N> x) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= x.length()) return;
        y(idx) = x(idx);
    }""")
    quote
        @assert length(y) == length(x)
        gdims, bdims = cudims(length(x))
        culaunch($f, gdims, bdims, y, x)
        y
    end
end

function Base.cat(dim::Int, xs::AbstractCuArray{T,N}...) where {T,N}
    cumdim = sum(x -> size(x,dim), xs)
    dims = ntuple(d -> d == dim ? cumdim : size(xs[1],d), N)
    y = CuArray{T}(dims)
    ysize = Any[Colon() for i=1:N]
    offset = 0
    for x in xs
        ysize[dim] = offset+1:offset+size(x,dim)
        suby = view(y, ysize...)
        copy!(suby, x)
        offset += size(x,dim)
    end
    y
end
