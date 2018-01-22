export CuStream

const CU_STREAM_DEFAULT      = Cint(0)
const CU_STREAM_NON_BLOCKING = Cint(1)

mutable struct CuStream
    ptr::Ptr{Void}

    function CuStream(flags::Cint=CU_STREAM_DEFAULT)
        ref = Ref{Ptr{Void}}()
        @apicall :cuStreamCreate (Ptr{Ptr{Void}},Cuint) ref flags
        s = new(ref[])
        finalizer(s, x -> @apicall :cuStreamDestroy (Ptr{Void},) x.ptr)
        s
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, s::CuStream) = s.ptr

synchronize(s::CuStream) = @apicall :cuStreamSynchronize (Ptr{Void},) s.ptr
