module NVRTC

if is_windows()
    const libnvrtc = Libdl.find_library(["nvrtc64_91","nvrtc64_90","nvrtc64_80","nvrtc64_75"])
else
    const libnvrtc = Libdl.find_library("libnvrtc")
end
const ACTIVE = !isempty(libnvrtc)

if ACTIVE
    ref_major = Ref{Cint}()
    ref_minor = Ref{Cint}()
    ccall((:nvrtcVersion,libnvrtc), Cint, (Ptr{Cint},Ptr{Cint}), ref_major, ref_minor)
    major = Int(ref_major[])
    minor = Int(ref_minor[])
    const API_VERSION = 1000major + 10minor
    info("NVRTC API $API_VERSION")
else
    const API_VERSION = 0
    warn("NVRTC library cannot be found.")
end

macro apicall(f, args...)
    quote
        result = ccall(($f,libnvrtc), Cint, $(map(esc,args)...))
        if result != 0
            Base.show_backtrace(STDOUT, backtrace())
            p = ccall((:nvrtcGetErrorString,libnvrtc), Cstring, (Cint,), result)
            throw(unsafe_string(p))
        end
    end
end

function compile(code::String; headers=[], include_names=[], options=[])
    ref = Ref{Ptr{Void}}()
    headers = Ptr{UInt8}[pointer(h) for h in headers]
    include_names = Ptr{UInt8}[pointer(n) for n in include_names]
    @apicall :nvrtcCreateProgram (Ptr{Ptr{Void}},Cstring,Cstring,Cint,Ptr{Ptr{UInt8}},Ptr{Ptr{UInt8}}) ref code C_NULL length(headers) headers include_names
    prog = ref[]

    options = Ptr{UInt8}[pointer(o) for o in options]
    try
        @apicall :nvrtcCompileProgram (Ptr{Void},Cint,Ptr{Ptr{UInt8}}) prog length(options) options
    catch
        ref = Ref{Csize_t}()
        @apicall :nvrtcGetProgramLogSize (Ptr{Void},Ptr{Csize_t}) prog ref
        log = Array{UInt8}(Int(ref[]))
        @apicall :nvrtcGetProgramLog (Ptr{Void},Ptr{UInt8}) prog log
        println()
        println("Error log:")
        println(String(log))
        throw("NVRTC compile failed.")
    end

    ref = Ref{Csize_t}()
    @apicall :nvrtcGetPTXSize (Ptr{Void},Ptr{Csize_t}) prog ref
    ptxsize = ref[]

    ptx = Array{UInt8}(ptxsize)
    @apicall :nvrtcGetPTX (Ptr{Void},Ptr{UInt8}) prog ptx
    @apicall :nvrtcDestroyProgram (Ptr{Ptr{Void}},) Ref{Ptr{Void}}(prog)

    String(ptx)
end

end
