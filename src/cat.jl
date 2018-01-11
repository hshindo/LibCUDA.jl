

@generated function nindex(i::Int, ls::NTuple{N}) where N
    quote
        Base.@_inline_meta
        $(foldr((n, els) -> :(i ≤ ls[$n] ? ($n, i) : (i -= ls[$n]; $els)), :(-1, -1), 1:N))
    end
end

function catindex(dim, I::NTuple{N}, shapes) where N
    @inbounds x, i = nindex(I[dim], getindex.(shapes, dim))
    x, ntuple(n -> n == dim ? i : I[n], Val{N})
end

function _cat(dim, dest, xs...)
    function kernel(dim, dest, xs)
        I = @cuindex dest
        n, I′ = catindex(dim, I, size.(xs))
        @inbounds dest[I...] = xs[n][I′...]
        return
    end
    println("_cat reached.")
    #blk, thr = cudims(dest)
    #@cuda (blk, thr) kernel(dim, dest, xs)
    dest
end

function Base.cat_t(dims::Int, T::Type, x::CuArray, xs::CuArray...)
    catdims = Base.dims2cat(dims)
    shape = Base.cat_shape(catdims, (), size.((x, xs...))...)
    dest = Base.cat_similar(x, T, shape)
    _cat(dims, dest, x, xs...)
end

Base.vcat(xs::CuArray...) = cat(1, xs...)
Base.hcat(xs::CuArray...) = cat(2, xs...)

macro cuindex(A)
    quote
        A = $(esc(A))
        i = (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
        i > length(A) && return
        ind2sub(A, i)
    end
end

function Base.fill!(xs::CuArray, x)
    function kernel(xs, x)
        I = @cuindex xs
        xs[I...] = x
        return
    end
    blk, thr = cudims(xs)
    @cuda (blk, thr) kernel(xs, convert(eltype(xs), x))
    return xs
end
