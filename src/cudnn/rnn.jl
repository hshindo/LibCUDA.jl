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

"""
    rnn()

T = Float32
x = curand(T, 10, 5)
"""
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
