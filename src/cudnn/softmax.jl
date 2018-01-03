function softmax(x::CuArray{T,N}, algo) where {T,N}
    @assert 1 <= N <= 4
    mode = CUDNN_SOFTMAX_MODE_CHANNEL
    xdesc = TensorDesc(x, 4)
    y = similar(x)

    @apicall(:cudnnSoftmaxForward,
        (Ptr{Void},Cint,Cint,
        Ptr{Void},Ptr{Void},Ptr{Void},
        Ptr{Void},Ptr{Void},Ptr{Void}),
        handle(), algo, mode, T[1], xdesc, x, T[0], xdesc, y)
    y, (xdesc,)
end

function ∇softmax!(y::CuArray{T}, dy, dx, xdesc, algo) where T
    mode = CUDNN_SOFTMAX_MODE_CHANNEL
    @apicall(:cudnnSoftmaxBackward,
        (Ptr{Void},Cint,Cint,
        Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},
        Ptr{Void},Ptr{Void},Ptr{Void}),
        handle(), algo, mode, T[1], ydesc, y, ydesc, dy, T[1], ydesc, dx)
end
