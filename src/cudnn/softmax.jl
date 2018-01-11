# cudnnSoftmaxAlgorithm_t
const CUDNN_SOFTMAX_FAST = Cint(0)
const CUDNN_SOFTMAX_ACCURATE = Cint(1)
const CUDNN_SOFTMAX_LOG = Cint(2)

# cudnnSoftmaxMode_t
const CUDNN_SOFTMAX_MODE_INSTANCE = Cint(0)
const CUDNN_SOFTMAX_MODE_CHANNEL = Cint(1)

function softmax(x::CuArray{T,N}, algo::Cint=CUDNN_SOFTMAX_ACCURATE) where {T,N}
    mode = CUDNN_SOFTMAX_MODE_CHANNEL
    xdesc = TensorDesc(x, 4)
    y = similar(x)
    @cudnn(:cudnnSoftmaxForward,
        (Cptr,Cint,Cint,
        Cptr,Cptr,Cptr,
        Cptr,Cptr,Cptr),
        gethandle(), algo, mode, T[1], xdesc, x, T[0], xdesc, y)
    y
end

function ∇softmax!(y::CuArray{T}, dy, dx, algo::Cint=CUDNN_SOFTMAX_ACCURATE) where T
    ydesc = TensorDesc(y, 4)
    mode = CUDNN_SOFTMAX_MODE_CHANNEL
    @cudnn(:cudnnSoftmaxBackward,
        (Cptr,Cint,Cint,
        Cptr,Cptr,Cptr,Cptr,Cptr,
        Cptr,Cptr,Cptr),
        gethandle(), algo, mode, T[1], ydesc, y, ydesc, dy, T[1], ydesc, dx)
end

logsoftmax(x::CuArray) = softmax(x, CUDNN_SOFTMAX_LOG)
∇logsoftmax!(y::CuArray, dy, dx) = ∇softmax!(y, dy, dx, CUDNN_SOFTMAX_LOG)
