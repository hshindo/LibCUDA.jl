struct NaiveAllocator
end

function alloc(::NaiveAllocator, bytesize::Int)
    @assert bytesize >= 0
    bytesize == 0 && return MemBlock(UInt64(0),0)

    dptr = memalloc_gc(bytesize)
    ptr = MemBlock(dptr, bytesize)
    finalizer(ptr, x -> memfree(x.dptr))
    ptr
end

free(::NaiveAllocator) = nothing
