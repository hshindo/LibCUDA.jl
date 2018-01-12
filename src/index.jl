@generated function Base.getindex(x::CuArray{T}, inds::Tuple) where T
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

getindex(x::Var, inds::Tuple) = Var(x.data[inds...], (getindex,x,inds))
getindex(x::Var, inds...) = getindex(x, inds)
