const FreeCuPtrs = [Dict{Int,Vector{UInt64}}() for i=1:ndevices()]
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

    buffers = FreeCuPtrs[dev+1]
    buffer = get!(buffers,bufid) do
        Ptr{UInt64}[]
    end
    if isempty(buffer)
        ref = Ref{Ptr{Void}}()
        @apicall :cuMemAlloc (Ptr{Ptr{Void}},Csize_t) ref bytesize
        ptr = ref[]
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
