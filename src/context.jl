mutable struct CuContext
    ptr::Ptr{Void}

    function CuContext(dev::Int)
        ref = Ref{Ptr{Void}}()
        @apicall :cuCtxCreate (Ptr{Ptr{Void}},Cuint,Cint) ref 0 dev
        ctx = new(ref[])
        # finalize(ctx, x -> @apicall :cuCtxDestroy (Ptr{Void},) x.ptr)
        ctx
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, ctx::CuContext) = ctx.ptr

const CUCONTEXTS = Any[nothing for _=1:nthreads()]
