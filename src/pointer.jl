const FreeCuPtrs = [Dict{Int,Vector{UInt64}}() for i=1:ndevices()]

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

# not tested
function devicegc()
    gc()
    _dev = getdevice()
    for dev = 0:ndevices()-1
        setdevice(dev)
        for (id,ptrs) in freeptrs[dev+1]
            for p in ptrs
                cuMemFree(p)
            end
            empty!(ptrs)
        end
    end
    setdevice(_dev)
    run(`nvidia-smi`)
end
