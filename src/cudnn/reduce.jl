mutable struct ReduceTensorDesc
    ptr::Ptr{Void}
end

function ReduceTensorDesc(::Type{T}, op::Cint) where T
    ref = Ref{Ptr{Void}}()
    @apicall :cudnnCreateReduceTensorDescriptor (Ptr{Ptr{Void}},) ref
    desc = ReduceTensorDesc(ref[])
    finalizer(desc, x -> @apicall :cudnnDestroyReduceTensorDescriptor (Ptr{Void},) x)

    ind = op == CUDNN_REDUCE_TENSOR_MIN || op == CUDNN_REDUCE_TENSOR_MAX ?
        CUDNN_REDUCE_TENSOR_FLATTENED_INDICES :
        CUDNN_REDUCE_TENSOR_NO_INDICES
    @apicall(:cudnnSetReduceTensorDescriptor,
        (Ptr{Void},Cint,Cint,Cint,Cint,Cint),
        desc, op, datatype(T), CUDNN_NOT_PROPAGATE_NAN, ind, CUDNN.CUDNN_32BIT_INDICES)
    desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ReduceTensorDesc) = desc.ptr

function reduce(op, A::CuArray{T}) where T
    h = handle()
    reducedesc = ReduceTensorDesc(T, op)
    adesc = TensorDesc(A, 4)
    cdims = ntuple(ndims(A)) do i
        i == dim ? 1 : size(A,i)
    end
    C = similar(A, cdims)
    cdesc = TensorDesc(C, 4)

    ref = Ref{Csize_t}()
    @apicall :cudnnGetReductionIndicesSize (Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Csize_t}) h reducedesc adesc cdesc ref
    indices = CuArray{UInt8}(Int(ref[]))

    ref = Ref{Csize_t}()
    @apicall :cudnnGetReductionWorkspaceSize (Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Csize_t}) h reducedesc adesc cdesc ref
    workspace = CuArray{UInt8}(Int(ref[]))

    @apicall(:cudnnReduceTensor,
        (Ptr{Void},Ptr{Void},Ptr{Void},Csize_t,Ptr{Void},Csize_t,
        Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void}),
        h, reducedesc, indices, length(indices), workspace, length(workspace),
        [one(T)], adesc, A, [zero(T)], cdesc, C)
end
