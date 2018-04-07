mutable struct CuPtr
    ptr::Ptr{Void}
    bytesize::Int
    ctx::CuContext

    function CuPtr(bytesize::Int)
        bytesize == 0 && return new(C_NULL,0,CuContext(C_NULL))
        ref = Ref{Ptr{Void}}()
        @apicall :cuMemAlloc (Ptr{Ptr{Void}},Csize_t) ref bytesize
        ctx = getcontext()
        ptr = new(ref[], bytesize, ctx)
        # finalizer(ptr, memfree)
        ptr
    end
end

Base.convert(::Type{Ptr{T}}, p::CuPtr) where T = Ptr{T}(p.ptr)
Base.unsafe_convert(::Type{Ptr{T}}, p::CuPtr) where T = Ptr{T}(p.ptr)

function memfree(ptr::CuPtr)
    setcontext(ptr.ctx) do
        @apicall :cuMemFree (Ptr{Void},) ptr
    end
end

function meminfo()
    ref_free = Ref{Csize_t}()
    ref_total = Ref{Csize_t}()
    @apicall(:cuMemGetInfo, (Ptr{Csize_t},Ptr{Csize_t}), ref_free, ref_total)
    Int(ref_free[]), Int(ref_total[])
end

function memalloc_gc(bytesize::Int)
    ref = Ref{UInt64}()
    status = @apicall_nocheck :cuMemAlloc (Ptr{UInt64},Csize_t) ref bytesize
    if status != CUDA_SUCCESS
        gc(false)
        status = @apicall_nocheck :cuMemAlloc (Ptr{UInt64},Csize_t) ref bytesize
        if status != CUDA_SUCCESS
            gc()
            @apicall :cuMemAlloc (Ptr{UInt64},Csize_t) ref bytesize
        end
    end
    ref[]
end
