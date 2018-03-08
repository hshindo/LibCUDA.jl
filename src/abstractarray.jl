export AbstractCuArray, AbstractCuVector, AbstractCuMatrx, AbstractCuVecOrMat
abstract type AbstractCuArray{T,N} end

const AbstractCuVector{T} = AbstractCuArray{T,1}
const AbstractCuMatrix{T} = AbstractCuArray{T,2}
const AbstractCuVecOrMat{T} = Union{AbstractCuVector{T},AbstractCuMatrix{T}}

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
