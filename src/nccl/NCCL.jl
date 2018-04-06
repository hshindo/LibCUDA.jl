module NCCL

using ..LibCUDA

if is_windows()
else
    const libnccl = Libdl.find_library("libnccl")
end
isempty(libnccl) && error("NCCL cannot be found.")

const NCCL_UNIQUE_ID_BYTES = 128

function init()
    global const API_VERSION = Int(ccall((:cudnnGetVersion,libcudnn),Cint,()))
    info("NCCL API $API_VERSION")
end
init()

macro nccl(f, args...)
    quote
        result = ccall(($f,libnccl), Cint, $(map(esc,args)...))
        if result != 0
            p = ccall((:ncclGetErrorString,libnccl), Ptr{UInt8}, (Cint,), result)
            throw(unsafe_string(p))
        end
    end
end

include("comm.jl")

end
