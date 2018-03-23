module NVML

using ..LibCUDA

if is_windows()
    const libnvml = Libdl.find_library("nvml", [joinpath(ENV["ProgramFiles"],"NVIDIA Corporation","NVSMI")])
else
    const libnvml = Libdl.find_library("libnvml")
end
const ACTIVE = !isempty(libnvml)

function checkresult(result::Cint)
    if result != 0
        p = ccall((:nvmlErrorString,libnvml), Ptr{Cchar}, (Cint,), ref)
        throw(unsafe_string(p))
    end
end

if ACTIVE
    result = ccall((:nvmlInit_v2,libnvml), Cint, ())
    checkresult(result)

    ref = Array{Cchar}(80)
    result = ccall((:nvmlSystemGetNVMLVersion,libnvml), Cint, (Ptr{Cchar},Cuint), ref, 80)
    checkresult(result)
    const API_VERSION = unsafe_string(pointer(ref))
    info("NVML $API_VERSION")
else
    warn("NVML API cannot be found.")
end

include("define.jl")

macro apicall(f, args...)
    f = get(DEFINE, f.args[1], f.args[1])
    if ACTIVE
        quote
            result = ccall(($(QuoteNode(f)),libnvml), Cint, $(map(esc,args)...))
            checkresult(result)
        end
    end
end

macro apicall_nocheck(f, args...)
    f = get(DEFINE, f.args[1], f.args[1])
    if ACTIVE
        quote
            ccall(($(QuoteNode(f)),libnvml), Cint, $(map(esc,args)...))
        end
    end
end

include("device.jl")

end
