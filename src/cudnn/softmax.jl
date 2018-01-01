function _softmax(algo, x::CuArray{T,N}) where {T,N}
    @assert 1 <= N <= 4
    mode = CUDNN_SOFTMAX_MODE_CHANNEL
    xdesc = TensorDesc(x, 4)
    y = similar(x)

    @apicall(:cudnnSoftmaxForward,
        (Ptr{Void},Cint,Cint,
        Ptr{Void},Ptr{Void},Ptr{Void},
        Ptr{Void},Ptr{Void},Ptr{Void}),
        handle(), algo, mode, [T(1)], xdesc, x, [T(0)], xdesc, y)
    xdesc, x, y
end
softmax(x::CuArray) = _softmax(CUDNN_SOFTMAX_ACCURATE,x)[3]
logsoftmax(x::CuArray) = _softmax(CUDNN_SOFTMAX_LOG,x)[3]

function ∇softmax!(algo, ydesc, y::CuArray{T}, dy, dx) where T
    mode = CUDNN_SOFTMAX_MODE_CHANNEL
    @apicall(:cudnnSoftmaxBackward,
        (Ptr{Void},Cint,Cint,
        Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},
        Ptr{Void},Ptr{Void},Ptr{Void}),
        handle(), algo, mode, [T(1)], ydesc, y, ydesc, dy, [T(1)], ydesc, dx)
end
