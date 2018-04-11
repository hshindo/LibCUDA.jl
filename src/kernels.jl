@generated function Base.copy!(dest::AbstractCuArray{T,N}, src::AbstractCuArray{T,N}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    $Array_h
    __global__ void copy(Array<$Ct,$N> dest, Array<$Ct,$N> src) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= src.length()) return;
        dest(idx) = src(idx);
    }""")
    quote
        @assert length(dest) == length(src)
        gdims, bdims = cudims(length(src))
        $k(gdims, bdims, DeviceArray(dest), DeviceArray(src))
        dest
    end
end

@generated function Base.fill!(x::AbstractCuArray{T,N}, value) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    $Array_h
    __global__ void fill(Array<$Ct,$N> x, $Ct value) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= x.length()) return;
        x(idx) = value;
    }""")
    quote
        gdims, bdims = cudims(length(x))
        $k(gdims, bdims, DeviceArray(x), T(value))
        x
    end
end
