module LibCUDA

if is_windows()
    const libcuda = Libdl.find_library(["nvcuda"])
else
    const libcuda = Libdl.find_library(["libcuda"])
end
if isempty(libcuda)
    warn("CUDA library cannot be found.")
end

ccall((:cuInit,libcuda), Cint, (Cint,), 0)

const API_VERSION = begin
    ref = Ref{Cint}()
    ccall((:cuDriverGetVersion,libcuda), Cint, (Ptr{Cint},), ref)
    Int(ref[])
end
info("CUDA API $API_VERSION")

include("define.jl")

macro apicall(f, args...)
    f = get(define, f.args[1], f.args[1])
    quote
        status = ccall(($(QuoteNode(f)),libcuda), Cint, $(map(esc,args)...))
        if status != 0
            Base.show_backtrace(STDOUT, backtrace())
            ref = Ref{Cstring}()
            ccall((:cuGetErrorString,libcuda), Cint, (Cint,Ptr{Cstring}), status, ref)
            throw(unsafe_string(ref[]))
        end
    end
end

include("device.jl")
include("context.jl")
include("stream.jl")
include("pointer.jl")
include("module.jl")
include("function.jl")
include("execution.jl")
include("interop.jl")

include("NVRTC.jl")
include("array.jl")
include("arraymath.jl")
include("cublas/CUBLAS.jl")
# include("cudnn/CUDNN.jl")

using .CUBLAS

const CuContexts = Ptr{Void}[]

function init(dev::Int)
    info("Initializing device $dev...")

    while length(CuContexts) < dev + 1
        push!(CuContexts, Ptr{Void}(0))
    end
    if CuContexts[dev+1] == Ptr{Void}(0)
        ref = Ref{Ptr{Void}}()
        @apicall :cuCtxCreate (Ptr{Ptr{Void}},Cuint,Cint) ref 0 dev
        ctx = ref[]
        CuContexts[dev+1] = ctx
        atexit(() -> @apicall :cuCtxDestroy (Ptr{Void},) ctx)
    end
    setdevice(dev)
    cap = capability(dev)
    mem = round(Int, totalmem(dev) / (1024^2))
    info("device[$dev]: $(devicename(dev)), capability $(cap[1]).$(cap[2]), totalmem = $(mem) MB")
end
init(0)

end
