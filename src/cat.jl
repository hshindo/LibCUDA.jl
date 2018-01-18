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
