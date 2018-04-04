function Base.cat(dim::Int, xs::AbstractCuArray{T}...) where T
    length(xs) == 1 && return xs[1]
    N = max(dim, maximum(ndims,xs))
    dims = Int[size(xs[1],i) for i=1:N]
    xs = map(xs) do x
        if ndims(x) == N
            x
        else
            dims[dim] = size(x,dim)
            reshape(x, dims...)
        end
    end

    dims[dim] = 0
    for x in xs
        dims[dim] += size(x,dim)
        for d = 1:N
            d == dim && continue
            @assert size(x,d) == size(xs[1],d)
        end
    end

    y = CuArray{T}(dims...)
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

function catlinear!(y::CuArray{T}, xs::CuArray{T}...) where T
    offset = 1
    for x in xs
        copy!(y, offset, x, 1, length(x))
        offset += length(x)
    end
    y
end

@generated function concat!(y::CuArray{T,N}, dim::Int, x1::CuArray{T,N}, x2::CuArray{T,N}) where {T,N}
    Ct = cstring(T)
    f = CuFunction("""
    $Array_h
    __global__ void concat(Array<$Ct,$N> y, int dim, $Ct *x1, int sizeX1, $Ct *x2, int sizeX2) {
        int idxY = blockIdx.x * blockDim.x + threadIdx.x;
        if (idxY >= y.length()) return;

        int ndIdx[$N];
        y.idx2ndIdx(ndIdx, idxY);
        $Ct *x;
        int sizeX;
        if (ndIdx[dim] < sizeX1) {
            x = x1;
            sizeX = sizeX1;
        }
        else {
            x = x2;
            sizeX = sizeX2;
            ndIdx[dim] -= sizeX1;
        }

        int idxX = 0;
        int strideX = 1;
        for (int d = 0; d < $N; d++) {
            idxX += ndIdx[d] * strideX;
            if (d == dim) strideX *= sizeX;
            else strideX *= y.dims[d];
        }
        y[idxY] = x[idxX];
    }""")
    quote
        gdims, bdims = cudims(length(y))
        culaunch($f, gdims, bdims, y, dim-1, x1.ptr, size(x1,dim), x2.ptr, size(x2,dim))
        y
    end
end

function cubox_cat(xs::Vector{CuArray{T,N}}) where {T,N}
    x = map(x -> x.ptr.dptr, xs)
    y = CuArray{UInt64}(length(xs))
    copy!(y, x)
end

@generated function concat_binarysearch!(y::CuArray{T,N}, dim::Int, xs::Vector{CuArray{T,N}}) where {T,N}
    Ct = cstring(T)
    f = CuFunction("""
    $Array_h

    __global__ void concat(Array<$Ct,$N> y, int dim, $Ct** xs, int lengthXs, int *cumdims) {
        int idxY = blockIdx.x * blockDim.x + threadIdx.x;
        if (idxY >= y.length()) return;

        int ndIdx[$N];
        y.idx2ndIdx(ndIdx, idxY);

        int left = 0;
        int right = lengthXs;
        while (left < right - 1) {
            int m = (left + right) / 2;
            if (ndIdx[dim] < cumdims[m]) right = m;
            else left = m;
        }

        int xsIdx = left;
        ndIdx[dim] -= cumdims[xsIdx];

        // get element of x
        int idxX = 0;
        int strideX = 1;
        for (int d = 0; d < $N; d++) {
            idxX += ndIdx[d] * strideX;
            if (d == dim) strideX *= cumdims[xsIdx+1] - cumdims[xsIdx];
            else strideX *= y.dims[d];
        }
        y[idxY] = xs[xsIdx][idxX];
    }""")
    quote
        cumdims = Array{Cint}(length(xs)+1)
        cumdims[1] = 0
        for i = 2:length(cumdims)
            cumdims[i] = cumdims[i-1] + size(xs[i-1],dim)
        end
        d_cumdims = CuArray(cumdims)
        d_xs = cubox(xs)

        gdims, bdims = cudims(length(y))
        culaunch($f, gdims, bdims, y, dim-1, d_xs.ptr, length(xs), d_cumdims.ptr)
        synchronize()
        y
    end
end
