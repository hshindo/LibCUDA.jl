const Array_h = open(readstring, joinpath(@__DIR__,"device/array.h"))

struct CuDeviceArray{T,N}
    ptr::Ptr{T}
    dims::NTuple{N,Cint}
end

cubox(x::CuArray{T}) where T = CuDeviceArray(Ptr{T}(x), map(Cint,size(x)))
# cubox(x::CuSubArray{T}) where T = CuDeviceArray(Ptr{T}(x), map(Cint,size(x)), map(Cint,strides(x)))

@generated function Base.fill!(x::CuArray{T}, value; stream=C_NULL) where T
    Ct = cstring(T)
    f = CuFunction("""
    __global__ void f($Ct *x, int length, $Ct value) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) x[idx] = value;
    }""")
    quote
        gdims, bdims = cudims(length(x))
        culaunch($f, gdims, bdims, x.ptr, length(x), T(value), stream=stream)
        x
    end
end

function Base.cat(dim::Int, xs::CuArray{T,N}...) where {T,N}
    @assert length(xs) == 2
    cumdim = sum(x -> size(x,dim), xs)
    dims = ntuple(d -> d == dim ? cumdim : size(xs[1],d), N)
    y = CuArray{T}(dims)
    cat!(y, dim, xs...)
end

@generated function cat!(y::CuArray{T,N}, dim::Int, x1::CuArray{T,N}, x2::CuArray{T,N}) where {T,N}
    Ct = cstring(T)
    f = CuFunction("""
    $Array_h
    __global__ void f(Array<$Ct,$N> y, int dim, $Ct *x1, int sizeX1, $Ct *x2, int sizeX2) {
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

@generated function ∇concat!(y::CuArray{T,N}, dim::Int, x1::CuArray{T,N}, x2::CuArray{T,N}) where {T,N}
    Ct = cstring(T)
    f = CuFunction("""
    $Array_h
    __global__ void f(Array<$Ct,$N> y, int dim, $Ct *x1, int sizeX1, $Ct *x2, int sizeX2) {
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
        x[idxX] += y[idxY];
    }""")
    quote
        gdims, bdims = cudims(length(y))
        culaunch($f, gdims, bdims, y, dim-1, x1.ptr, size(x1,dim), x2.ptr, size(x2,dim))
        y
    end
end

@generated function concat2!(y::CuArray{T,N}, dim::Int, xs::Vector{CuArray{T,N}}) where {T,N}
    Ct = cstring(T)
    f = CuFunction("""
    $Array_h

    __global__ void f(Array<$Ct,$N> y, int dim, $Ct *xs[], int lengthXs, int *cumdims) {
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

        int idxX = 0;
        int strideX = 1;
        for (int d = 0; d < $N; d++) {
            idxX += ndIdx[d] * strideX;
            //if (d == dim) strideX *= cumdims[xsIdx+1] - cumdims[xsIdx];
            //else strideX *= y.dims[d];
        }
        //$Ct* t = reinterpret_cast<$Ct*>(xs[0]);
        //if (idxY == 22) y[0] = ($Ct)xs[0];
    }""")
    quote
        cumdims = Array{Cint}(length(xs)+1)
        cumdims[1] = 0
        for i = 2:length(cumdims)
            cumdims[i] = cumdims[i-1] + size(xs[i-1],dim)
        end
        println(cumdims)

        ptr_xs = cubox2(xs)
        ptr_cumdims = CuArray(cumdims).ptr
        gdims, bdims = cudims(length(y))
        culaunch($f, gdims, bdims, y, dim-1, ptr_xs, length(xs), ptr_cumdims)
        synchronize()
        y
    end
end

cubox2(xs::Vector{CuArray{T,N}}) where {T,N} = map(x -> Ptr{Void}(x.ptr), xs)

@generated function maximum_batch!(x::CuArray{T,N}, dims::CuVector{Cint}) where {T,N}
    Ct = cstring(T)
    f = CuFunction("""
    $Array_h
    __global__ void f(Array<$Ct,$N> y, int dim, $Ct *xs[], int lengthXs, int *cumdims) {
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

        int idxX = 0;
        int strideX = 1;
        for (int d = 0; d < $N; d++) {
            idxX += ndIdx[d] * strideX;
            //if (d == dim) strideX *= cumdims[xsIdx+1] - cumdims[xsIdx];
            //else strideX *= y.dims[d];
        }
        //$Ct* t = reinterpret_cast<$Ct*>(xs[0]);
        //if (idxY == 22) y[0] = ($Ct)xs[0];
    }""")
    quote
        cumdims = Array{Cint}(length(xs)+1)
        cumdims[1] = 0
        for i = 2:length(cumdims)
            cumdims[i] = cumdims[i-1] + size(xs[i-1],dim)
        end
        println(cumdims)

        ptr_xs = cubox2(xs)
        ptr_cumdims = CuArray(cumdims).ptr
        gdims, bdims = cudims(length(y))
        culaunch($f, gdims, bdims, y, dim-1, ptr_xs, length(xs), ptr_cumdims)
        synchronize()
        y
    end
end
