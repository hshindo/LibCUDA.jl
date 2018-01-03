mutable struct FilterDesc
    ptr::Ptr{Void}
end

function FilterDesc{T,N}(x::CuArray{T,N}; format=CUDNN_TENSOR_NCHW)
    ref = Ref{Ptr{Void}}()
    @apicall :cudnnCreateFilterDescriptor (Ptr{Ptr{Void}},) ref
    desc = FilterDesc(ref[])
    finalizer(desc, x -> @apicall :cudnnDestroyFilterDescriptor (Ptr{Void},) x)

    csize = Cint[size(x,i) for i=N:-1:1]
    @apicall :cudnnSetFilterNdDescriptor () desc datatype(T) format N csize
    desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::FilterDesc) = desc.ptr
