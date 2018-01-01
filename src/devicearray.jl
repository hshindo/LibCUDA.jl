const Array_h = open(readstring, joinpath(@__DIR__,"device/array.h"))

struct CuDeviceArray{T,N}
    ptr::Ptr{T}
    dims::NTuple{N,Cint}
    strides::NTuple{N,Cint}
end

cubox(x::CuArray{T}) where T = CuDeviceArray(Ptr{T}(x), map(Cint,size(x)), map(Cint,strides(x)))
# cubox(x::CuSubArray{T}) where T = CuDeviceArray(Ptr{T}(x), map(Cint,size(x)), map(Cint,strides(x)))

@generated function test_array(x::CuMatrix{T}) where T
    Ct = cstring(T)
    f = CuFunction("""
    $Array_h

    __global__ void f(Array<$Ct,2> x, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            x(idx) = 3;
        }
    }""")
    quote
        gdims, bdims = cudims(length(x))
        culaunch($f, gdims, bdims, x, length(x))
        x
    end
end
