module LibCUDA

if is_windows()
    const libcuda = Libdl.find_library(["nvcuda"])
else
    const libcuda = Libdl.find_library(["libcuda"])
end
const Configured = !isempty(libcuda)
Configured || warn("CUDA library cannot be found. LibCUDA does not work correctly.")

if Configured
    ccall((:cuInit,libcuda), Cint, (Cint,), 0)
end

const API_VERSION = begin
    if Configured
        ref = Ref{Cint}()
        ccall((:cuDriverGetVersion,libcuda), Cint, (Ptr{Cint},), ref)
        Int(ref[])
    else
        -1
    end
end
Configured && info("CUDA API $API_VERSION")

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

CurrentDevice = -1

cstring(::Type{Float32}) = "float"
cstring(::Type{Int}) = "int"

include("device.jl")
include("memory.jl")
include("stream.jl")
include("pointer.jl")
include("module.jl")
include("function.jl")
include("execution.jl")
Configured && include("NVRTC.jl")

include("array.jl")
include("arraymath.jl")
# include("cat.jl")
include("devicearray.jl")

if Configured
    include("cublas/CUBLAS.jl")
    include("cudnn/CUDNN.jl")

    using .CUDNN
    export CUDNN
end

end
