mutable struct AtomicMalloc
    ptr::Ptr{Void}
    bytesize::Int
    offset::Int
    unused::Vector{UInt64}

    function AtomicMalloc()
        a = new(C_NULL, 0, 0, UInt64[])
        finalizer(a, dispose)
        a
    end
end

function (a::AtomicMalloc)(bytesize::Int)
    @assert bytesize >= 0
    bytesize == 0 && return MemBlock(UInt64(0),0)

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
    MemBlock(dptr, bytesize)
end

function free(a::AtomicMalloc)
    a.offset = 0
    foreach(memfree, a.unused)
    empty!(a.unused)
end

function dispose(x::AtomicMalloc)
    free(x)
    memfree(x.ptr)
end
