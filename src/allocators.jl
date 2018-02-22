struct NaiveAllocator
end

getallocator() = Allocators[1]

function setallocator(x)
    a = Allocators[1]
    free(a)
    Allocators[1] = x
end

function alloc(::NaiveAllocator, bytesize::Int)
    @assert bytesize >= 0
    bytesize == 0 && return CuPtr(UInt64(0),0)

    dptr = memalloc_gc(bytesize)
    ptr = CuPtr(dptr, bytesize)
    finalizer(ptr, x -> memfree(x.dptr))
    ptr
end

free(::NaiveAllocator) = nothing

doc"""
    GreedyAllocator
"""
mutable struct GreedyAllocator
    dptr::UInt64
    bytesize::Int
    offset::Int
    unused::Vector{UInt64}

    function GreedyAllocator()
        a = new(UInt64(0), 0, 0, UInt64[])
        finalizer(a, dispose)
        a
    end
end

function alloc(a::GreedyAllocator, bytesize::Int)
    @assert bytesize >= 0
    bytesize == 0 && return CuPtr(UInt64(0),0)

    if bytesize + a.offset > a.bytesize
        push!(a.unused, a.dptr)
        a.bytesize = bytesize + a.offset
        #while bytesize + mem.index > mem.bytesize
        #    mem.bytesize *= 2
        #end
        a.dptr = memalloc_gc(a.bytesize)
        a.offset = 0
    end
    dptr = a.dptr + a.offset
    a.offset += bytesize
    CuPtr(dptr, bytesize)
end

function free(a::GreedyAllocator)
    a.offset = 0
    foreach(memfree, a.unused)
    empty!(a.unused)
end

function dispose(a::GreedyAllocator)
    free(a)
    memfree(a.dptr)
end


mutable struct MemoryPool
    dptrs::Vector{Vector{UInt64}}

    function MemoryPool()
        x = new(Vector{UInt64}[])
        finalizer(x, dispose)
        x
    end
end

function dispose(x::MemoryPool)
    gc()
    for dptrs in x.dptrs
        for dptr in dptrs
            memfree(dptr)
        end
    end
end

function log2id(bytesize::Int)
    bufsize = bytesize - 1
    id = 1
    while bufsize > 1
        bufsize >>= 1
        id += 1
    end
    id
end

function alloc(mem::MemoryPool, bytesize::Int)
    @assert bytesize >= 0
    #dev = getdevice()
    #if dev < 0
    #    throw("GPU device is not set. Call `setdevice(dev)`.")
    #end
    bytesize == 0 && return CuPtr(UInt64(0),0)
    id = log2id(bytesize)
    bytesize = 1 << id

    while length(mem.dptrs) < id
        push!(mem.dptrs, CuPtr[])
    end
    buffer = mem.dptrs[id]
    if isempty(buffer)
        ref = Ref{UInt64}()
        status = @apicall_nocheck :cuMemAlloc (Ptr{UInt64},Csize_t) ref bytesize
        if status != CUDA_SUCCESS
            gc()
            if isempty(buffer)
                dispose(mem)
                dptr = memalloc(bytesize)
            else
                dptr = pop!(buffer)
            end
        else
            dptr = ref[]
        end
    else
        dptr = pop!(buffer)
    end
    ptr = CuPtr(dptr, bytesize)
    finalizer(ptr, x -> free(mem,x))
    ptr
end

function free(mem::MemoryPool, ptr::CuPtr)
    id = log2id(ptr.bytesize)
    push!(mem.dptrs[id], ptr.dptr)
end

const Allocators = Any[MemoryPool()]
