module CUBLAS

using ..LibCUDA

if is_windows()
    const libcublas = Libdl.find_library(["cublas64_91","cublas64_90","cublas64_80","cublas64_75"])
else
    const libcublas = Libdl.find_library(["libcublas"])
end
isempty(libcublas) && warn("CUBLAS library cannot be found.")

include("define.jl")

function errorstring(status)
    status == CUBLAS_STATUS_SUCCESS && return "SUCCESS"
    status == CUBLAS_STATUS_NOT_INITIALIZED && return "NOT_INITIALIZED"
    status == CUBLAS_STATUS_ALLOC_FAILED && return "ALLOC_FAILED"
    status == CUBLAS_STATUS_INVALID_VALUE && return "INVALID_VALUE"
    status == CUBLAS_STATUS_ARCH_MISMATCH && return "ARCH_MISMATCH"
    status == CUBLAS_STATUS_MAPPING_ERROR && return "MAPPING_ERROR"
    status == CUBLAS_STATUS_EXECUTION_FAILED && return "EXECUTION_FAILED"
    status == CUBLAS_STATUS_INTERNAL_ERROR && return "INTERNAL_ERROR"
    status == CUBLAS_STATUS_NOT_SUPPORTED && return "NOT_SUPPORTED"
    status == CUBLAS_STATUS_LICENSE_ERROR && return "LICENSE_ERROR"
    throw("UNKNOWN ERROR")
end

macro apicall(f, rettypes, args...)
    f = get(define, f.args[1], f.args[1])
    quote
        status = ccall(($(QuoteNode(f)),libcublas), Cint, $(esc(rettypes)), $(map(esc,args)...))
        if status != 0
            Base.show_backtrace(STDOUT, backtrace())
            throw(errorstring(status))
        end
    end
end

const Handles = Ptr{Void}[]

function handle()
    dev = getdevice()
    while length(Handles) < dev + 1
        push!(Handles, Ptr{Void}(0))
    end
    h = Handles[dev+1]
    if h == Ptr{Void}(0)
        ref = Ref{Ptr{Void}}()
        @apicall :cublasCreate (Ptr{Void},) ref
        h = ref[]
        Handles[dev+1] = h
        atexit(() -> @apicall :cublasDestroy (Ptr{Void},) h)
    end
    h
end

const API_VERSION = begin
    ref = Ref{Ptr{Void}}()
    @apicall :cublasCreate (Ptr{Void},) ref
    h = ref[]
    ref = Ref{Cint}()
    @apicall :cublasGetVersion (Ptr{Void},Ptr{Cint}) h ref
    @apicall :cublasDestroy (Ptr{Void},) h
    Int(ref[])
end
info("CUBLAS API $API_VERSION")

function cublasop(t::Char)
    t == 'N' && return Cint(0)
    t == 'T' && return Cint(1)
    t == 'C' && return Cint(2)
    throw("Unknown cublas operation: $(t).")
end

#include("level1.jl")
include("level2.jl")
include("level3.jl")

end
