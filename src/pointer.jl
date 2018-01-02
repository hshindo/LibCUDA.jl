const FreeCuPtrs = Dict{Int,Vector{UInt64}}[]
atexit() do
    gc()
    for d in FreeCuPtrs
        isempty(d) && continue
        for (k,v) in d
            for x in v
                @apicall :cuMemFree (UInt64,) x
            end
        end
    end
end

mutable struct CuPtr
    ptr::UInt64
    bufid::Int
    dev::Int
end

function CuPtr(bytesize::Int)
    @assert bytesize >= 0
    dev = getdevice()
    if dev < 0
        throw("GPU device is not set. Call `setdevice(dev)`.")
    end
    bytesize == 0 && return CuPtr(zero(UInt64),-1,dev)
    bufid = (bytesize-1) >> 10 + 1
    bytesize = bufid << 10

    while length(FreeCuPtrs) < dev + 1
        push!(FreeCuPtrs, Dict{Int,Vector{UInt64}}())
    end
    buffers = FreeCuPtrs[dev+1]
    buffer = get!(buffers,bufid) do
        Ptr{UInt64}[]
    end
    if isempty(buffer)
        ref = Ref{UInt64}()
        status = ccall((:cuMemAlloc_v2,libcuda), Cint, (Ptr{UInt64},Csize_t), ref, bytesize)
        if status != 0
            gc()
            if !isempty(buffer)
                ptr = pop!(buffer)
            else
                for (k,v) in buffers
                    for x in v
                        @apicall :cuMemFree (UInt64,) x
                    end
                end
                empty!(buffers)
                ptr = memalloc(bytesize)
            end
        else
            ptr = ref[]
        end
        #@apicall :cuMemAlloc (Ptr{Ptr{Void}},Csize_t) ref bytesize
        #ptr = ref[]
    else
        ptr = pop!(buffer)
    end
    cuptr = CuPtr(ptr, bufid, dev)
    finalizer(cuptr, x -> push!(FreeCuPtrs[x.dev+1][x.bufid], x.ptr))
    cuptr
end

Base.convert(::Type{Ptr{T}}, p::CuPtr) where T = Ptr{T}(p.ptr)
Base.convert(::Type{UInt64}, p::CuPtr) = p.ptr
Base.unsafe_convert(::Type{Ptr{T}}, p::CuPtr) where T = Ptr{T}(p.ptr)
Base.unsafe_convert(::Type{UInt64}, p::CuPtr) = p.ptr
