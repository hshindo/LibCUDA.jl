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
end

struct RNNWork
    seqlength
    xdesc
    hxdesc
    ydesc
    workspace
    reservespace
    t_x
    t_y
end

function RNN(insize::Int, hsize::Int, nlayers::Int, droprate::Float64, direction::Cint, mode::Cint, w::CuVector{T}) where T
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
    rnn = RNN(insize, hsize, nlayers, direction, desc, wdesc, w, zeros(w))
    finalizer(rnn, x -> @cudnn :cudnnDestroyRNNDescriptor (Cptr,) x.desc)
    rnn
end

function LSTM(insize, hsize, nlayers, droprate, w::CuVector)
    RNN(insize, hsize, nlayers, droprate, CUDNN_UNIDIRECTIONAL, CUDNN_LSTM, w)
end
function BiLSTM(insize, hsize, nlayers, droprate, w::CuVector)
    RNN(hsize, nlayers, droprate, CUDNN_BIDIRECTIONAL, CUDNN_LSTM, w)
end

function (rnn::RNN)(x::CuMatrix{T}, batchdims::Vector{Int}; training=true) where T
    @assert rnn.insize == size(x,1)
    insize, hsize, nlayers = rnn.insize, rnn.hsize, rnn.nlayers
    t_x, t_batchdims = batch_rnn(x, batchdims)
    seqlength = size(t_x, 2)

    # x: (1,X,B,T) where X = inputSize, B = miniBatch, T = seqLength
    # xDesc: Array of T (1,X,B) descriptors
    xdesc = map(t_batchdims) do d
        TensorDesc(T, 1, insize, d)
    end

    # hx,cx,hy,cy: (H,B,L) where H = hidden size, L = numLayers * (bidirectional ? 2 : 1)
    coef = rnn.direction == CUDNN_UNIDIRECTIONAL ? 1 : 2
    hxdesc = TensorDesc(T, hsize, t_batchdims[1], nlayers*coef)
    hx = hy = cx = cy = C_NULL
    #hx = CuArray{T}(hsize, batchdims[1])
    #hy = similar(hx)
    #cx = similar(hx)
    #cy = similar(hx)

    # y: (1,Y,B,T) where Y = hiddenSize * (bidirectional ? 2 : 1)
    # yDesc: Array of T (1,Y,B) descriptors
    t_y = CuArray{T}(hsize*coef*sum(batchdims))
    ydesc = map(t_batchdims) do d
        TensorDesc(T, 1, hsize*coef, d)
    end

    # workspace = getworkspace(rnn, seqlength)
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
            xdesc, t_x,
            hxdesc, C_NULL,
            hxdesc, C_NULL,
            rnn.wdesc, rnn.w,
            ydesc, t_y,
            hxdesc, C_NULL,
            hxdesc, C_NULL,
            workspace, length(workspace),
            reservespace, length(workspace))
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
            xdesc, t_x,
            hxdesc, hx,
            hxdesc, cx,
            rnn.wdesc, rnn.w,
            ydesc, t_y,
            hxdesc, hy,
            hxdesc, cy,
            workspace, length(workspace))
    end
    y = ∇batch_rnn(t_y, t_batchdims, batchdims)
    if training
        work = RNNWork(seqlength, xdesc, hxdesc, ydesc, workspace, reservespace, t_x, t_y)
        y, work
    else
        y, nothing
    end
end

function backward_data(rnn::RNN, dy::CuArray, batchdims::Vector{Int}, work::RNNWork)
    t_dx = similar(work.t_x)
    t_dy, _ = batch_rnn(dy, batchdims)
    @assert size(work.t_y) == size(t_dy)
    h = gethandle()
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
        work.ydesc, work.t_y,
        work.ydesc, t_dy,
        work.hxdesc, dhy,
        work.hxdesc, dcy,
        rnn.wdesc, rnn.w,
        work.hxdesc, hx,
        work.hxdesc, cx,
        work.xdesc, t_dx,
        work.hxdesc, dhx,
        work.hxdesc, dcx,
        work.workspace, length(work.workspace),
        work.reservespace, length(work.reservespace))

    dx = ∇batch_rnn(t_dx, batchdims)
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
        work.xdesc, work.t_x,
        work.hxdesc, hx,
        work.ydesc, work.t_y,
        work.workspace, length(work.workspace),
        rnn.wdesc, rnn.dw,
        work.reservespace, length(work.reservespace))
end

function batch_rnn2(x::CuMatrix{T}, batchdims::Vector{Int}) where T
    perm = sortperm(batchdims, rev=true)
    xs = split(x, 2, batchdims)
    xs = xs[perm]
    insize = size(x, 1)
    seqlength = batchdims[perm[1]]
    t_x = CuArray{T}(insize*length(batchdims), seqlength)
    for i = 1:length(xs)
        v = view(t_x, (i-1)*insize+1:i*insize, 1:batchdims[perm[i]])
        copy!(v, xs[i])
    end
    k = length(batchdims)
    t_batchdims = Int[]
    for t = 1:seqlength
        while batchdims[perm[k]] < t
            k -= 1
        end
        push!(t_batchdims, k)
    end
    t_x, t_batchdims
end

@generated function batch_rnn(x::CuMatrix{T}, batchdims_x::Vector{Int}) where T
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)
    __global__ void batch_rnn($Ct *y, Array<$Ct,2> x, int *cumdims, int seqlength) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= x.length()) return;

        int xj = idx / hsize;
        int xi = idx - xj * hsize;
        int j = xj / seqlength;
        int i = xj - j * seqlength;
        int yi = xi;
        int yj = cumdims[j] + i;
        if (yj < cumdims[j+1]) {
            y[yi+yj*hsize] = x[idx];
        }
    }""")
    quote
        k = length(batchdims_x)
        batchdims_y = Int[]
        for t = 1:batchdims_x[1]
            while batchdims_x[k] < t
                k -= 1
            end
            push!(batchdims_y, k)
        end

        y = CuArray{T}(length(batchdims_x)*size(x,1), batchdims_x[1])
        cumdims = Array{Cint}(size(y,2)+1)
        cumdims[1] = 0
        for i = 2:length(cumdims)
            cumdims[i] = cumdims[i-1] + batchdims_y[i-1]
        end

        gdims, bdims = cudims(length(x))
        culaunch($f, gdims, bdims, y.ptr, x.ptr, CuArray(cumdims).ptr, length(x), size(x,1), length(y))
        y, batchdims_y
    end
end

@generated function ∇batch_rnn(y::CuMatrix{T}, batchdims_y::Vector{Int}) where T
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)
    __global__ void batch_rnn_grad($Ct *y, $Ct *x, int *cumdims, int hsize, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= length) return;

        int xj = idx / hsize;
        int xi = idx - xj * hsize;
        int yj = xj / cumdims[1];
        int yi = xj - yj * cumdims[1];
        if (cumdims[yj] + yi < cumdims[yj+1]) {
            int a = (yi + yj*cumdims[1]) * hsize + xi;
            int b = (cumdims[yj] + yi) * hsize + xi;
            y[a] = x[b];
        }
    }""")
    quote
        k = length(batchdims_x)
        batchdims_y = Int[]
        for t = 1:batchdims_x[1]
            while batchdims_x[k] < t
                k -= 1
            end
            push!(batchdims_y, k)
        end

        y = CuArray{T}(length(batchdims_x)*size(x,1), batchdims_x[1])
        cumdims = Array{Cint}(size(y,2)+1)
        cumdims[1] = 0
        for i = 2:length(cumdims)
            cumdims[i] = cumdims[i-1] + batchdims_y[i-1]
        end

        gdims, bdims = cudims(length(y))
        culaunch($f, gdims, bdims, y.ptr, x.ptr, CuArray(cumdims).ptr, size(x,1), length(y))
        y
    end
end

function ∇batch_rnn(t_x::CuVector{T}, t_batchdims::Vector{Int}, batchdims::Vector{Int}) where T
    for t = 1:10

    end

    perm = sortperm(batchdims, rev=true)
    hsize = size(t_x,1) ÷ length(batchdims)
    xs = Array{CuSubMatrix{T}}(length(batchdims))
    for i = 1:length(batchdims)
        xs[perm[i]] = view(t_x, (i-1)*hsize+1:i*hsize, 1:batchdims[perm[i]])
    end
    cat(2, xs...)
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
