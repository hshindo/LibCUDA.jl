mutable struct TensorDesc
    ptr::Ptr{Void}

    function TensorDesc(x::CuArray{T,N}, nd=ndims(x)) where {T,N}
        csize, cstrides = tensorsize(x, nd)
        ref = Ref{Ptr{Void}}()
        @apicall :cudnnCreateTensorDescriptor (Ptr{Ptr{Void}},) ref
        desc = new(ref[])
        finalizer(desc, x -> @apicall :cudnnDestroyTensorDescriptor (Ptr{Void},) x)
        @apicall :cudnnSetTensorNdDescriptor (Ptr{Void},Cint,Cint,Ptr{Cint},Ptr{Cint}) desc datatype(T) length(csize) csize cstrides
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::TensorDesc) = desc.ptr

function tensorsize(a, dims)
    sz = Cint[reverse(size(a))...]
    st = Cint[reverse(strides(a))...]
    if length(sz) == 1 < dims
        unshift!(sz, 1)
        unshift!(st, 1)
    end
    while length(sz) < dims
        push!(sz, 1)
        push!(st, 1)
    end
    while length(sz) > dims
        d = pop!(sz)
        sz[length(sz)] *= d
        pop!(st)
        st[length(st)] = 1
    end
    (sz, st)
end
