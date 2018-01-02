module CUDNN

using ..LibCUDA

if is_windows()
    const libcudnn = Libdl.find_library(["cudnn64_7"])
else
    const libcudnn = Libdl.find_library(["libcudnn"])
end
isempty(libcudnn) && throw("CUDNN library cannot be found.")

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

const Handles = Ptr{Void}[]

function handle()
    dev = getdevice()
    while length(Handles) < dev + 1
        push!(Handles, Ptr{Void}(0))
    end
    h = Handles[dev+1]
    if h == Ptr{Void}(0)
        ref = Ref{Ptr{Void}}()
        @apicall :cudnnCreate (Ptr{Ptr{Void}},) ref
        h = ref[]
        Handles[dev+1] = h
        atexit(() -> @apicall :cudnnDestroy (Ptr{Void},) h)
    end
    h
end

include("activation.jl")
include("dropout.jl")
include("reduce.jl")
include("rnn.jl")
include("softmax.jl")
include("tensor.jl")

end
