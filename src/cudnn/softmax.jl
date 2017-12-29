function softmax(algo, x::CuArray{T,N}) where {T,N}
    @assert 1 <= N <= 4
    h = handle()
    mode = CUDNN_SOFTMAX_MODE_CHANNEL
    xdesc = TensorDesc(x, 4)
    y = similar(x)

    @apicall(:cudnnSoftmaxForward,
        (Ptr{Void},Cint,Cint,
        Ptr{Void},Ptr{Void},Ptr{Void},
        Ptr{Void},Ptr{Void},Ptr{Void}),
        h, algo, mode, [one(T)], xdesc, x, [zero(T)], xdesc, y)
    xdesc, x, y
end

function ∇softmax!(algo, ydesc, y::CuArray{T}, dy, dx) where T
    h = handle()
    mode = CUDNN_SOFTMAX_MODE_CHANNEL
    @apicall(:cudnnSoftmaxBackward,
        (Ptr{Void},Cint,Cint,
        Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},
        Ptr{Void},Ptr{Void},Ptr{Void}),
        h, algo, mode, [one(T)], ydesc, y, ydesc, dy, [one(T)], ydesc, dx)
end
