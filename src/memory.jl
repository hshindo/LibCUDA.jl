function memalloc(bytesize::Int)
    ref = Ref{UInt64}()
    @apicall :cuMemAlloc (Ptr{UInt64},Csize_t) ref bytesize
    ref[]
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

function memfree(dptr::UInt64)
    @apicall :cuMemFree (UInt64,) dptr
end
