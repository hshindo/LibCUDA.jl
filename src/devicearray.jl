const Array_h = open(readstring, joinpath(@__DIR__,"device/array.h"))

struct CuDeviceArray{T,N}
    ptr::Ptr{T}
    dims::NTuple{N,Cint}
    strides::NTuple{N,Cint}
    contigious::Cuchar
end

cubox(x::CuArray{T}) where T = CuDeviceArray(Ptr{T}(x), map(Cint,size(x)), map(Cint,strides(x)), Cuchar(1))
cubox(x::CuSubArray{T}) where T = CuDeviceArray(Ptr{T}(x), map(Cint,size(x)), map(Cint,strides(x)), Cuchar(0))
