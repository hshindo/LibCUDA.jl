module LibCUDA

using Base.Threads

if is_windows()
    const libcuda = Libdl.find_library("nvcuda")
else
    const libcuda = Libdl.find_library("libcuda")
end
const AVAILABLE = !isempty(libcuda)
AVAILABLE || warn("CUDA cannot be found.")

function checkstatus(status)
    if status != 0
        ref = Ref{Cstring}()
        ccall((:cuGetErrorString,libcuda), Cint, (Cint,Ptr{Cstring}), status, ref)
        throw(unsafe_string(ref[]))
    end
end

function init()
    if AVAILABLE
        status = ccall((:cuInit,libcuda), Cint, (Cint,), 0)
        checkstatus(status)

        ref = Ref{Cint}()
        status = ccall((:cuDriverGetVersion,libcuda), Cint, (Ptr{Cint},), ref)
        checkstatus(status)
        
        global const API_VERSION = Int(ref[])
        info("CUDA API $API_VERSION")
    else
        global const API_VERSION = 0
    end
end
init()

include("define.jl")

macro apicall(f, args...)
    f = get(define, f.args[1], f.args[1])
    quote
        status = ccall(($(QuoteNode(f)),libcuda), Cint, $(map(esc,args)...))
        checkstatus(status)
    end
end

macro apicall_nocheck(f, args...)
    f = get(define, f.args[1], f.args[1])
    quote
        ccall(($(QuoteNode(f)),libcuda), Cint, $(map(esc,args)...))
    end
end

include("driver/device.jl")
include("driver/context.jl")
include("driver/stream.jl")
include("driver/memory.jl")
include("driver/allocators.jl")
include("driver/module.jl")
include("driver/function.jl")

# This must be loaded before kernel.jl and kernels.jl
if AVAILABLE
    include("nvml/NVML.jl")
    include("NVRTC.jl")
    using .NVML
end

include("abstractarray.jl")
include("array.jl")
include("subarray.jl")
include("lineararray.jl")
include("kernel.jl")
include("kernels.jl")

include("arraymath.jl")
include("cat.jl")
include("reduce.jl")
include("reducedim.jl")
include("devicearray.jl")

if AVAILABLE
    include("nccl/NCCL.jl")
    include("cublas/CUBLAS.jl")
    include("cudnn/CUDNN.jl")

    using .CUBLAS, .CUDNN
    export CUBLAS, CUDNN
end

end
