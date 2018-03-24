module CUDNN

using ..LibCUDA

if is_windows()
    const libcudnn = Libdl.find_library(["cudnn64_7"])
else
    const libcudnn = Libdl.find_library("libcudnn")
end
isempty(libcudnn) && error("CUDNN cannot be found.")

function init()
    global const API_VERSION = Int(ccall((:cudnnGetVersion,libcudnn),Cint,()))
    info("CUDNN API $API_VERSION")
end
init()

macro cudnn(f, args...)
    quote
        status = ccall(($f,libcudnn), Cint, $(map(esc,args)...))
        if status != 0
            p = ccall((:cudnnGetErrorString,libcudnn), Ptr{UInt8}, (Cint,), status)
            throw(unsafe_string(p))
        end
    end
end

include("define.jl")

# cudnnDataType_t
const CUDNN_DATA_FLOAT = Cint(0)
const CUDNN_DATA_DOUBLE = Cint(1)
const CUDNN_DATA_HALF = Cint(2)
const CUDNN_DATA_INT8 = Cint(3)
const CUDNN_DATA_INT32 = Cint(4)
const CUDNN_DATA_INT8x4 = Cint(5)

const Cptr = Ptr{Void}
datatype(::Type{Float32}) = CUDNN_DATA_FLOAT
datatype(::Type{Float64}) = CUDNN_DATA_DOUBLE
datatype(::Type{Float16}) = CUDNN_DATA_HALF
datatype(::Type{Int8}) = CUDNN_DATA_INT8
datatype(::Type{Int32}) = CUDNN_DATA_INT32

mutable struct Handle
    ptr::Ptr{Void}

    function Handle()
        ref = Ref{Ptr{Void}}()
        @cudnn :cudnnCreate (Ptr{Ptr{Void}},) ref
        h = new(ref[])
        # @cudnn :cudnnDestroy (Ptr{Void},) h)
        h
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, h::Handle) = h.ptr

const HANDLES = Array{Handle}(ndevices())

function gethandle()
    dev = getdevice()
    isassigned(HANDLES,dev) || (HANDLES[dev+1] = Handle())
    HANDLES[dev+1]
end

function setstream(handle::Handle, stream)
    @cudnn :cudnnSetStream (Ptr{Void},Ptr{Void}) handle stream
end

include("activation.jl")
include("convolution.jl")
include("filter.jl")
include("dropout.jl")
include("reduce.jl")
include("rnn.jl")
include("softmax.jl")
include("tensor.jl")

"""
C = α*A + β*C

The bias tensor A must match the corresponding dimension of the destination tensor
C or must be equal to 1.
"""
function add!(α, A::CuArray{T}, β, C::CuArray{T}) where T
    h = gethandle()
    adesc = TensorDesc(A, 4)
    cdesc = TensorDesc(C, 4)
    @cudnn(:cudnnAddTensor,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr),
        h, T[α], adesc, A, T[β], cdesc, C)
    C
end

end
