export cstring

cstring(::Type{Int32}) = "int"
cstring(::Type{Float32}) = "float"

@generated function Base.copy!(dest::AbstractCuArray{T,N}, src::AbstractCuArray{T,N}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void copy(Array<$Ct,$N> dest, Array<$Ct,$N> src) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= src.length()) return;

        int ndIdx[$N];
        dest.idx2ndIdx(ndIdx, idx);
        dest(ndIdx) = src(ndIdx);
    }""")
    quote
        @assert length(dest) == length(src)
        gdims, bdims = cudims(length(src))
        $k(gdims, bdims, dest, src)
        dest
    end
end

@generated function Base.fill!(x::CuLinearArray{T}, value) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void fill($Ct *x, int n, $Ct value) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        x[idx] = value;
    }""")
    quote
        gdims, bdims = cudims(length(x))
        $k(gdims, bdims, pointer(x), length(x), T(value))
        x
    end
end

@generated function Base.fill!(x::AbstractCuArray{T,N}, value) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void fill(Array<$Ct,$N> x, $Ct value) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= x.length()) return;

        x(idx) = value;
    }""")
    quote
        gdims, bdims = cudims(length(x))
        $k(gdims, bdims, x, T(value))
        x
    end
end
