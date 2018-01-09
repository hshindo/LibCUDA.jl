# cudnnTensorFormat_t
const CUDNN_TENSOR_NCHW = Cint(0)
const CUDNN_TENSOR_NHWC = Cint(1)
const CUDNN_TENSOR_NCHW_VECT_C = Cint(2)

mutable struct TensorDesc
    ptr::Cptr

    function TensorDesc(::Type{T}, csize::Vector{Cint}, cstrides::Vector{Cint}) where T
        ref = Ref{Cptr}()
        @apicall :cudnnCreateTensorDescriptor (Ptr{Cptr},) ref
        desc = new(ref[])
        finalizer(desc, destroy)

        @apicall(:cudnnSetTensorNdDescriptor,
            (Cptr,Cint,Cint,Ptr{Cint},Ptr{Cint}),
            desc, datatype(T), length(csize), csize, cstrides)
        desc
    end
end

destroy(desc::TensorDesc) = @apicall :cudnnDestroyTensorDescriptor (Cptr,) desc.ptr

function TensorDesc(x::CuArray{T}, nd=ndims(x)) where T
    ndims(x) == 1 && (x = reshape(x,length(x),1))
    csize = Cint[reverse(size(x))...]
    cstrides = Cint[reverse(strides(x))...]
    while length(csize) < nd
        push!(csize, 1)
        push!(cstrides, 1)
    end
    while length(csize) > nd
        d = pop!(csize)
        csize[length(csize)] *= d
        pop!(cstrides)
        cstrides[length(cstrides)] = 1
    end
    TensorDesc(T, csize, cstrides)
end

function TensorDesc(::Type{T}, dims::Union{Tuple,Vector}) where T
    length(dims) == 1 && (dims = [dims[1],1])
    strides = Array{Cint}(length(dims))
    strides[1] = 1
    for i = 1:length(strides)-1
        strides[i+1] = strides[i] * dims[i]
    end
    csize = Cint[dims[i] for i=length(dims):-1:1]
    cstrides = reverse(strides)
    TensorDesc(T, csize, cstrides)
end
TensorDesc(::Type{T}, dims::Int...) where T = TensorDesc(T, dims)

Base.convert(::Type{Cptr}, desc::TensorDesc) = desc.ptr
Base.unsafe_convert(::Type{Cptr}, desc::TensorDesc) = desc.ptr

#=
function tensorsize(a, dims)
    sz = Cint[reverse(size(a))...]
    st = Cint[reverse(strides(a))...]
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
=#
