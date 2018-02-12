# cudnnRNNMode_t
const CUDNN_RNN_RELU = Cint(0)
const CUDNN_RNN_TANH = Cint(1)
const CUDNN_LSTM = Cint(2)
const CUDNN_GRU = Cint(3)

# cudnnDirectionMode_t
const CUDNN_UNIDIRECTIONAL = Cint(0)
const CUDNN_BIDIRECTIONAL = Cint(1)

# cudnnRNNInputMode_t
const CUDNN_LINEAR_INPUT = Cint(0)
const CUDNN_SKIP_INPUT = Cint(1)

# cudnnRNNAlgo_t
const CUDNN_RNN_ALGO_STANDARD = Cint(0)
const CUDNN_RNN_ALGO_PERSIST_STATIC = Cint(1)
const CUDNN_RNN_ALGO_PERSIST_DYNAMIC = Cint(2)

mutable struct RNNDesc
    ptr::Cptr

    function RNNDesc(::Type{T}, hsize::Int, nlayers::Int, droprate::Float64, direction, mode) where T
        ref = Ref{Cptr}()
        @cudnn :cudnnCreateRNNDescriptor (Ptr{Cptr},) ref
        desc = new(ref[])
        finalizer(desc, x -> @cudnn :cudnnDestroyRNNDescriptor (Cptr,) x.ptr)

        h = gethandle()
        dropdesc = DropoutDesc(droprate)
        algo = CUDNN_RNN_ALGO_STANDARD
        @cudnn(:cudnnSetRNNDescriptor,
            (Cptr,Cptr,Cint,Cint,Cptr,Cint,Cint,Cint,Cint,Cint),
            h, desc, hsize, nlayers, dropdesc, CUDNN_LINEAR_INPUT, direction, mode, algo, datatype(T))
        desc
    end
end

Base.unsafe_convert(::Type{Cptr}, desc::RNNDesc) = desc.ptr

### Size chart (Julia sizes for CUDNN calls)
# Note: For Julia calls, x and y do not need the initial 1 dimension and B,T are optional.
#
# x: (1,X,B,T) where X = inputSize, B = miniBatch, T = seqLength
# xDesc: Array of T (1,X,B) descriptors
# y: (1,Y,B,T) where Y = hiddenSize * (bidirectional ? 2 : 1)
# yDesc: Array of T (1,Y,B) descriptors
# w: (1,1,W) where W = cudnnGetRNNParamsSize()
# hx,cx,hy,cy: (H,B,L) where H = hidden size, L = numLayers * (bidirectional ? 2 : 1)
#
# Note: cudnn docs say min tensor dims 4 but RNN_example.cu uses 3D tensors
mutable struct RNN
    insize::Int
    hsize::Int
    nlayers::Int
    direction::Cint
    desc::Cptr
    wdesc::FilterDesc
    w::CuVector
    dw::CuVector
    hx::CuVector
    hxdesc::FilterDesc
    dhx::CuVector
    cx::CuVector
    dcx::CuVector
end

struct RNNWork
    seqlength
    xdesc
    hxdesc
    ydesc
    workspace
    reservespace
    x
    y
end

function RNN(insize::Int, hsize::Int, nlayers::Int, droprate::Float64, direction::Cint, mode::Cint, w::CuVector{T}, hx::CuVector{T}, cx::CuVector{T}) where T
    ref = Ref{Cptr}()
    @cudnn :cudnnCreateRNNDescriptor (Ptr{Cptr},) ref
    desc = ref[]

    h = gethandle()
    dropdesc = DropoutDesc(droprate)
    algo = CUDNN_RNN_ALGO_STANDARD
    @cudnn(:cudnnSetRNNDescriptor,
        (Cptr,Cptr,Cint,Cint,Cptr,Cint,Cint,Cint,Cint,Cint),
        h, desc, hsize, nlayers, dropdesc, CUDNN_LINEAR_INPUT, direction, mode, algo, datatype(T))

    wdesc = FilterDesc(T, 1, 1, length(w))
    rnn = RNN(insize, hsize, nlayers, direction, desc, wdesc, w, zeros(w), hx, zeros(hx), cx, zeros(cx))
    finalizer(rnn, x -> @cudnn :cudnnDestroyRNNDescriptor (Cptr,) x.desc)
    rnn
end

function (rnn::RNN)(x::CuMatrix{T}, batchdims::Vector{Int}; training=true) where T
    @assert rnn.insize == size(x,1)
    insize, hsize, nlayers = rnn.insize, rnn.hsize, rnn.nlayers
    seqlength = length(batchdims)

    # x: (1,X,B,T) where X = inputSize, B = miniBatch, T = seqLength
    # xDesc: Array of T (1,X,B) descriptors
    xdesc = map(batchdims) do d
        TensorDesc(T, 1, insize, d)
    end

    # hx,cx,hy,cy: (H,B,L) where H = hidden size, L = numLayers * (bidirectional ? 2 : 1)
    coef = rnn.direction == CUDNN_UNIDIRECTIONAL ? 1 : 2
    hxdesc = TensorDesc(T, hsize, batchdims[1], nlayers*coef)
    hy = zeros(rnn.hx)
    cy = zeros(rnn.cx)

    # y: (1,Y,B,T) where Y = hiddenSize * (bidirectional ? 2 : 1)
    # yDesc: Array of T (1,Y,B) descriptors
    y = CuArray{T}(hsize*coef, sum(batchdims))
    ydesc = map(batchdims) do d
        TensorDesc(T, 1, hsize*coef, d)
    end

    h = gethandle()
    ref = Ref{Csize_t}()
    @cudnn(:cudnnGetRNNWorkspaceSize,
        (Cptr,Cptr,Cint,Ptr{Cptr},Ptr{Csize_t}),
        h, rnn.desc, seqlength, xdesc, ref)
    workspace = CuArray{UInt8}(Int(ref[]))

    if training
        ref = Ref{Csize_t}()
        @cudnn(:cudnnGetRNNTrainingReserveSize,
            (Cptr,Cptr,Cint,Ptr{Cptr},Ptr{Csize_t}),
            h, rnn.desc, seqlength, xdesc, ref)
        reservespace = CuArray{UInt8}(Int(ref[]))

        @cudnn(:cudnnRNNForwardTraining,
            (Cptr,Cptr,Cint,
            Ptr{Cptr},Cptr,     # x
            Cptr,Cptr,          # hx
            Cptr,Cptr,          # cx
            Cptr,Cptr,          # w
            Ptr{Cptr},Cptr,     # y
            Cptr,Cptr,          # hy
            Cptr,Cptr,          # cy
            Cptr,Csize_t,       # workspace
            Cptr,Csize_t),      # reservespace
            h, rnn.desc, seqlength,
            xdesc, x,
            hxdesc, rnn.hx,
            hxdesc, rnn.cx,
            rnn.wdesc, rnn.w,
            ydesc, y,
            hxdesc, hy,
            hxdesc, cy,
            workspace, length(workspace),
            reservespace, length(workspace))
        work = RNNWork(seqlength, xdesc, hxdesc, ydesc, workspace, reservespace, x, y)
        y, work
    else
        @cudnn(:cudnnRNNForwardInference,
            (Cptr,Cptr,Cint,
            Ptr{Cptr},Cptr,     # x
            Cptr,Cptr,          # hx
            Cptr,Cptr,          # cx
            Cptr,Cptr,          # w
            Ptr{Cptr},Cptr,     # y
            Cptr,Cptr,          # hy
            Cptr,Cptr,          # cy
            Cptr,Csize_t),      # workspace
            h, rnn.desc, seqlength,
            xdesc, x,
            hxdesc, rnn.hx,
            hxdesc, rnn.cx,
            rnn.wdesc, rnn.w,
            ydesc, y,
            hxdesc, hy,
            hxdesc, cy,
            workspace, length(workspace))
        y, nothing
    end
end

function backward_data(rnn::RNN, dy::CuArray, work::RNNWork)
    h = gethandle()
    dx = similar(work.x)
    dhy = dcy = hx = cx = dhx = dcx = C_NULL
    @cudnn(:cudnnRNNBackwardData,
        (Cptr,Cptr,Cint,
        Ptr{Cptr},Cptr,     # y
        Ptr{Cptr},Cptr,     # dy
        Cptr,Cptr,  # dhy
        Cptr,Cptr,  # dcy
        Cptr,Cptr,  # w
        Cptr,Cptr,  # hx
        Cptr,Cptr,  # cx
        Ptr{Cptr},Cptr,  # dx
        Cptr,Cptr,  # dhx
        Cptr,Cptr,  # dcx
        Cptr,Csize_t,   # workspace
        Cptr,Csize_t),  # reservespace
        h, rnn.desc, work.seqlength,
        work.ydesc, work.y,
        work.ydesc, dy,
        work.hxdesc, dhy,
        work.hxdesc, dcy,
        rnn.wdesc, rnn.w,
        work.hxdesc, hx,
        work.hxdesc, cx,
        work.xdesc, dx,
        work.hxdesc, dhx,
        work.hxdesc, dcx,
        work.workspace, length(work.workspace),
        work.reservespace, length(work.reservespace))
    dx
end

function backward_weights!(rnn::RNN, work::RNNWork)
    h = gethandle()
    hx = C_NULL
    @cudnn(:cudnnRNNBackwardWeights,
        (Cptr,Cptr,Cint,
        Ptr{Cptr},Cptr,     # x
        Cptr,Cptr,          # hx
        Ptr{Cptr},Cptr,     # y
        Cptr,Csize_t,       # workspace
        Cptr,Cptr,          # dw
        Cptr,Csize_t),      # reservespace
        h, rnn.desc, work.seqlength,
        work.xdesc, work.x,
        work.hxdesc, hx,
        work.ydesc, work.y,
        work.workspace, length(work.workspace),
        rnn.wdesc, rnn.dw,
        work.reservespace, length(work.reservespace))
end

function split(x::CuArray{T,N}, dim::Int, splitsize::Vector{Int}) where {T,N}
    dims = Any[Colon() for i=1:N]
    offset = 0
    map(splitsize) do s
        dims[dim] = offset+1:offset+s
        offset += s
        view(x, dims...)
    end
end

function getRNNParamSize(::Type{T}, desc, xdesc) where T
    h = gethandle()
    ref = Ref{Csize_t}()
    @cudnn(:cudnnGetRNNParamsSize,
        (Cptr,Cptr,Cptr,Ptr{Csize_t},Cint),
        h, desc, xdesc, ref, datatype(T))
    println(Int(ref[]) ÷ sizeof(T))
end
