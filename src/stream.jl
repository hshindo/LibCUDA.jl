mutable struct CuStream
    ptr::Ptr{Void}

    function CuStream(flags::Int=0)
        ref = Ref{Ptr{Void}}()
        @apicall :cuStreamCreate (Ptr{Ptr{Void}},Cuint) ref flags
        s = new(ref[])
        finalizer(s, x -> @apicall :cuStreamDestroy (Ptr{Void},) x)
        s
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, s::CuStream) = s.ptr

#const CuDefaultStream() = CuStream()

synchronize(s::CuStream) = @apicall :cuStreamSynchronize (Ptr{Void},) s
