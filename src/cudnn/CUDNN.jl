module CUDNN

using ..LibCUDA

if is_windows()
    const libcudnn = Libdl.find_library(["cudnn64_7"])
else
    const libcudnn = Libdl.find_library(["libcudnn"])
end
if isempty(libcudnn)
    warn("CUDNN library cannot be found.")
end

const API_VERSION = Int(ccall((:cudnnGetVersion,libcudnn),Cint,()))
info("CUDNN API $API_VERSION")

macro apicall(f, args...)
    quote
        status = ccall(($f,libcudnn), Cint, $(map(esc,args)...))
        if status != 0
            Base.show_backtrace(STDOUT, backtrace())
            p = ccall((:cudnnGetErrorString,libcudnn), Ptr{UInt8}, (Cint,), status)
            throw(unsafe_string(p))
        end
    end
end

include("define.jl")

datatype(::Type{Float32}) = CUDNN_DATA_FLOAT
datatype(::Type{Float64}) = CUDNN_DATA_DOUBLE
datatype(::Type{Float16}) = CUDNN_DATA_HALF
datatype(::Type{Int8}) = CUDNN_DATA_INT8
datatype(::Type{Int32}) = CUDNN_DATA_INT32

const Handles = cudnnHandle_t[]
handle(x::CuArray) = Handles[getdevice(x)+1]

function init()
    empty!(Handles)
    ref = Ref{Ptr{Void}}()
    @apicall :cudnnCreate (Ptr{Ptr{Void}},) ref
    h = ref[]
    push!(Handles, h)
    atexit() do
        @apicall :cudnnDestroy (Ptr{Void},) h
    end
end
init()

mutable struct ActivationDesc
    ptr::Ptr{Void}

    function ActivationDesc()
        ref = Ref{Ptr{Void}}()
        @apicall :cudnnCreateActivationDescriptor (Ptr{Ptr{Void}},) ref
        desc = new(ref[])
        finalizer(desc, x -> @apicall :cudnnDestroyActivationDescriptor (Ptr{Void},) x)
        desc
    end
end
Base.unsafe_convert(::Type{Ptr{Void}}, desc::ActivationDesc) = desc.ptr

mutable struct TensorDesc
    ptr::Ptr{Void}

    function TensorDesc(x::CuArray{T,N}; pad=0) where {T,N}
        csize = Cint[1, 1, 1, 1]
        cstrides = Cint[1, 1, 1, 1]
        st = strides(x)
        for i = 1:N
            csize[4-i-pad+1] = size(x,i)
            cstrides[4-i-pad+1] = st[i]
        end
        ref = Ref{Ptr{Void}}()
        @apicall :cudnnCreateTensorDescriptor (Ptr{Ptr{Void}}) ref
        cudnnSetTensorNdDescriptor(ref[], datatype(T), length(csize), csize, cstrides)
        desc = new(ref[])
        finalizer(desc, x -> @apicall :cudnnDestroyTensorDescriptor (Ptr{Void},) x)
        desc
    end
end
Base.unsafe_convert(::Type{Ptr{Void}}, desc::TensorDesc) = desc.ptr

#Base.unsafe_convert(::Type{Ptr{Void}}, desc::DropoutDesc) = desc.ptr
#Base.unsafe_convert(::Type{Ptr{Void}}, desc::ReduceTensorDesc) = desc.ptr
#Base.unsafe_convert(::Type{Ptr{Void}}, desc::RNNDesc) = desc.ptr
#Base.unsafe_convert(::Type{Ptr{Void}}, desc::FilterDesc) = desc.ptr

end
