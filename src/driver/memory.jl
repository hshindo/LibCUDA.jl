export CuPtr

mutable struct CuPtr{T}
    ptr::Ptr{T}
    n::Int
    ctx::CuContext

    function CuPtr{T}(n::Int) where T
        n == 0 && return new(Ptr{T}(0),0,CuContext(C_NULL))
        ref = Ref{Ptr{Void}}()
        @apicall :cuMemAlloc (Ptr{Ptr{Void}},Csize_t) ref n*sizeof(T)
        ctx = getcontext()
        ptr = new(Ptr{T}(ref[]), n, ctx)
        # finalizer(ptr, memfree)
        ptr
    end
end

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
