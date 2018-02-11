module LibCUDA

if is_windows()
    const libcuda = Libdl.find_library(["nvcuda"])
else
    const libcuda = Libdl.find_library(["libcuda"])
end

const Configured = !isempty(libcuda)

function checkstatus(status)
    if status != 0
        # Base.show_backtrace(STDOUT, backtrace())
        ref = Ref{Cstring}()
        ccall((:cuGetErrorString,libcuda), Cint, (Cint,Ptr{Cstring}), status, ref)
        throw(unsafe_string(ref[]))
    end
end

if Configured
    status = ccall((:cuInit,libcuda), Cint, (Cint,), 0)
    checkstatus(status)

    ref = Ref{Cint}()
    ccall((:cuDriverGetVersion,libcuda), Cint, (Ptr{Cint},), ref)
    const API_VERSION = Int(ref[])
    info("CUDA API $API_VERSION")
else
    warn("CUDA library cannot be found. LibCUDA does not work correctly.")
    const API_VERSION = 0
end

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

export cstring
cstring(::Type{Float32}) = "float"

include("device.jl")
include("pointer.jl")
include("stream.jl")
include("memory.jl")
include("allocators.jl")
include("module.jl")
include("function.jl")
include("execution.jl")
Configured && include("NVRTC.jl")

include("abstractarray.jl")
include("array.jl")
include("arraymath.jl")
include("subarray.jl")
include("cat.jl")
include("devicearray.jl")
include("reduce.jl")
include("reducedim.jl")

if Configured
    setdevice(0)
    include("cublas/CUBLAS.jl")
    include("cudnn/CUDNN.jl")

    using .CUDNN
    export CUDNN
end

end
