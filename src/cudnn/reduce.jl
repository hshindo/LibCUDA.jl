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

function reduce(A::CuArray{T}, dim, op) where T
    h = handle()
    reducedesc = ReduceTensorDesc(T, op)
    adesc = TensorDesc(A, 4)
    cdims = ntuple(ndims(A)) do i
        i == dim ? 1 : size(A,i)
    end
    C = similar(A, cdims)
    cdesc = TensorDesc(C, 4)

    ref = Ref{Csize_t}()
    @apicall(:cudnnGetReductionIndicesSize,
        (Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Csize_t}),
        h, reducedesc, adesc, cdesc, ref)
    indices = CuArray{Cint}(Int(ref[])÷sizeof(Cint))

    ref = Ref{Csize_t}()
    @apicall(:cudnnGetReductionWorkspaceSize,
        (Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Csize_t}),
        h, reducedesc, adesc, cdesc, ref)
    workspace = CuArray{UInt8}(Int(ref[]))

    @apicall(:cudnnReduceTensor,
        (Ptr{Void},Ptr{Void},Ptr{Void},Csize_t,Ptr{Void},Csize_t,
        Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void}),
        h, reducedesc, indices, length(indices)*sizeof(Cint), workspace, length(workspace),
        [T(1)], adesc, A, [T(0)], cdesc, C)

    isempty(indices) ? C : (C,indices)
end

Base.sum(x::CuArray, dim::Int) = reduce(x, dim, CUDNN_REDUCE_TENSOR_ADD)
mul(x::CuArray, dim) = reduce(x, dim, CUDNN_REDUCE_TENSOR_MUL)
Base.findmax(x::CuArray, dim) = reduce(x, dim, CUDNN_REDUCE_TENSOR_MAX)
Base.findmin(x::CuArray, dim) = reduce(x, dim, CUDNN_REDUCE_TENSOR_MIN)
Base.maximum(::typeof(abs), x::CuArray, dim::Int) = reduce(x, dim, CUDNN_REDUCE_TENSOR_AMAX)
Base.mean(x::CuArray, dim) = reduce(x, dim, CUDNN_REDUCE_TENSOR_AVG)
function Base.norm(x::CuArray, dim::Int, p::Int)
    if p == 1
        reduce(x, dim, CUDNN_REDUCE_TENSOR_NORM1)
    elseif p == 2
        reduce(x, dim, CUDNN_REDUCE_TENSOR_NORM2)
    else
        throw("Not supported. Valid p: 1 or 2.")
    end
end
# mul_nozeros(x::CuArray, dim) = reduce(x, dim, CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS)
