function memalloc(bytesize::Int)
    ref = Ref{UInt64}()
    @apicall :cuMemAlloc (Ptr{UInt64},Csize_t) ref bytesize
    ref[]
end

function memfree(dptr::UInt64)
    @apicall :cuMemFree (UInt64,) dptr
end

#=
mutable struct MemoryBuffer
    ptr::UInt64
    bytesize::Int
    index::Int
end
MemoryBuffer() = MemoryBuffer(UInt64(0), 0, 0)

const MemBuffers = [MemoryBuffer()]

function cumalloc(bytesize::Int)
    dev = getdevice()
    mem = MemBuffers[dev+1]
    if mem.bytesize == 0 || bytesize + mem.index > mem.bytesize
        mem.ptr = memalloc(bytesize)
        mem.bytesize = bytesize
        mem.index = 0
    else
        ptr = mem.ptr + mem.index
        mem.index += bytesize
    end
    ptr
end

function cufree()
    dev = getdevice()
    mem = MemBuffers[dev+1]
    mem.index = 0
end
=#
