export MemBlock

mutable struct MemBlock{T}
    ptr::Ptr{T}
    n::Int
    dev::Int
end

function MemBlock(::Type{T}, n::Int) where T
    n == 0 && return MemBlock(Ptr{T}(0),0,-1)
    ref = Ref{Ptr{Void}}()
    @apicall :cuMemAlloc (Ptr{Ptr{Void}},Csize_t) ref n*sizeof(T)
    dev = getdevice()
    mb = MemBlock(Ptr{T}(ref[]), n, dev)
    finalizer(mb, memfree)
    mb
end

Base.convert(::Type{Ptr{T}}, x::MemBlock) where T = Ptr{T}(x.ptr)
Base.unsafe_convert(::Type{Ptr{T}}, x::MemBlock) where T = Ptr{T}(x.ptr)

function memfree(x::MemBlock)
    setdevice(x.dev) do
        @apicall :cuMemFree (Ptr{Void},) x.ptr
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
