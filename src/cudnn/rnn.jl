mutable struct RNNDesc
    ptr::Ptr{Void}
end

function RNNDesc(::Type{T}, hsize::Int, nlayers::Int, droprate::Float64, dir, algo) where T
    ref = Ref{Ptr{Void}}()
    @apicall :cudnnCreateRNNDescriptor (Ptr{Ptr{Void}},) ref
    desc = RNNDesc(ref[])
    finalizer(desc, x -> @apicall :cudnnDestroyRNNDescriptor (Ptr{Void},) x)

    dropdesc = DropoutDesc(droprate)
    @apicall(:cudnnSetRNNDescriptor,
        (Ptr{Void},Cint,Cint,Ptr{Void},Cint,Cint,Cint,Cint,Cint),
        desc, hsize, nlayers, dropdesc, CUDNN.CUDNN_LINEAR_INPUT, dir, mode, algo, datatype(T))
    desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::RNNDesc) = desc.ptr

function rnn(x::CuArray{T}, sizes::Vector, hsize::Int, nlayers::Int, droprate::Float64, dir, algo) where T
    @assert issorted(sizes, by=length, rev=true)
    xdesc = Ptr{Void}[TensorDesc(s) for s in sizes]

    rnndesc = RNNDesc(T, hsize, nlayers, droprate, dir, algo)
    h = handle()
    seqlength = sizes[1][end]
    ref = Ref{Csize_t}()
    @apicall :cudnnGetRNNWorkspaceSize (Ptr{Void},Ptr{Void},Cint,Ptr{Ptr{Void}},Ptr{Csize_t}) h rnndesc seqlength xdesc ref
    workspace = CuArray{UInt8}(Int(ref[]))

    ref = Ref{Csize_t}()
    @apicall :cudnnGetRNNTrainingReserveSize (Ptr{Void},Ptr{Void},Cint,Ptr{Ptr{Void}},Ptr{Csize_t}) h rnndesc seqlength xdesc ref
    reservespace = CuArray{UInt8}(Int(ref[]))

    ref = Ref{Csize_t}()
    @apicall :cudnnGetRNNParamsSize (Ptr{Void},Ptr{Void},Ptr{Ptr{Void}},Ptr{Csize_t},Cint) h rnndesc xdesc ref datatype(T)
    params = CuArray{UInt8}(Int(ref[]))

    linLayerMatDesc = FilterDesc()
    ref = Ref{Ptr{Void}}()
    #@apicall :cudnnGetRNNLinLayerMatrixParams () h rnndesc layer xdesc wdesc w linLayerID linLayerMatDesc ref
    linLayerMatDesc = ref[]

    #cudnnGetRNNLinLayerBiasParams()

    #=
    @apicall(:cudnnRNNForwardTraining,
        (Ptr{Void},Ptr{Void},Cint,
        Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},
        Ptr{Void},Ptr{Void},
        Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},
        Ptr{Void},Csize_t,Ptr{Void})
        h, rnndesc, seqlength,
        xdesc, x, hxdesc, hx, cxdesc, cx,
        wdesc, w,
        ydesc, y, hydesc, hy, cydesc, cy,
        workspace, length(workspace), reservespace)
    =#
end

export
    # cudnnRNNInputMode_t
    CUDNN_LINEAR_INPUT,
    CUDNN_SKIP_INPUT,

    # cudnnDirectionMode_t
    CUDNN_UNIDIRECTIONAL,
    CUDNN_BIDIRECTIONAL,

    # cudnnRNNMode_t
    CUDNN_RNN_RELU,
    CUDNN_RNN_TANH,
    CUDNN_LSTM,
    CUDNN_GRU

    function rnn_training!{T}(xdims, x::CuArray{T}, hx::CuArray{T}, cx::CuArray{T},
        droprate, input_t, dir_t, net_t; seed=0)

        xdesc = tensor_desc(CuArray(T, xdims[1]...))
        xdescs = fill(xdesc, length(xdims))
        for i=1:length(xdims) xdescs[i] = tensor_desc(CuArray(T, xdims[i])) end
        hxdesc = tensor_desc(hx)
        cxdesc = tensor_desc(cx)

        y = similar(x)
        hy = similar(hx)
        cy = similar(cx)
        ydescs = similar(xdescs)
        for i=1:length(xdims) ydescs[i] = tensor_desc(CuArray(T, xdims[i])) end
        hydesc = tensor_desc(hy)
        cydesc = tensor_desc(cy)

        h = gethandle(device(x))
        rnndesc, dropdesc, dropstate = rnn_desc(x, size(hx,2), size(hx,4), input_t,
            dir_t, net_t, droprate, seed)
        wsize_p = Cint[0]
        cudnnGetRNNParamsSize(h, rnndesc, xdesc, wsize_p, datatype(T))
        wsize = wsize_p[1]
        w = curand(T, 1, 1, 1, Int(wsize/(T.size)))
        wdesc = filter_desc(w)

        worksize_p = Cint[0]
        cudnnGetRNNWorkspaceSize(h, rnndesc, Cint(length(xdescs)), xdescs, worksize_p)
        worksize = worksize_p[1]
        workspace = CuArray(Int8, Int(worksize))

        trainsize_p = Cint[0]
        cudnnGetRNNTrainingReserveSize(h, rnndesc, Cint(length(xdescs)), xdescs, trainsize_p)
        trainsize = trainsize_p[1]
        trainspace = CuArray(Int8, Int(trainsize))

        mdesc_p = Ptr{Void}[0]
        cudnnCreateFilterDescriptor(mdesc_p)
        mdesc = mdesc_p[1]
        m_p = Ptr{Void}[0]
        cudnnGetRNNLinLayerMatrixParams(h, rnndesc, Cint(0), xdesc, wdesc, w,
            Cint(0), mdesc, m_p)
        m = m_p[1]

        bdesc_p = Ptr{Void}[0]
        cudnnCreateFilterDescriptor(bdesc_p)
        bdesc = bdesc_p[1]
        b_p = Ptr{Void}[0]
        cudnnGetRNNLinLayerBiasParams(h, rnndesc, Cint(0), xdesc, wdesc, w,
            Cint(0), bdesc, b_p)
        b = b_p[1]

        cudnnRNNForwardTraining(h, rnndesc, Cint(length(xdescs)), xdescs, x, hxdesc,
            hx, cxdesc, cx, wdesc, w, ydescs, y, hydesc, hy, cydesc, cy, workspace,
            worksize, trainspace, trainsize)

        cudnnDestroyFilterDescriptor(bdesc)
        cudnnDestroyFilterDescriptor(mdesc)
        cudnnDestroyFilterDescriptor(wdesc)
        cudnnDestroyRNNDescriptor(rnndesc)
        cudnnDestroyTensorDescriptor(cydesc)
        cudnnDestroyTensorDescriptor(hydesc)
        for desc in ydescs cudnnDestroyTensorDescriptor(desc) end
        cudnnDestroyTensorDescriptor(cxdesc)
        cudnnDestroyTensorDescriptor(hxdesc)
        for desc in xdescs cudnnDestroyTensorDescriptor(desc) end
        cudnnDestroyTensorDescriptor(xdesc)
        w, y, hy, cy, dropdesc, dropstate
    end
