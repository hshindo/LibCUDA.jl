mutable struct ActivationDesc
    ptr::Ptr{Void}
end

function ActivationDesc(mode, coef::Float64=0.0)
    ref = Ref{Ptr{Void}}()
    @apicall :cudnnCreateActivationDescriptor (Ptr{Ptr{Void}},) ref
    desc = ActivationDesc(ref[])
    finalizer(desc, x -> @apicall :cudnnDestroyActivationDescriptor (Ptr{Void},) x)

    @apicall :cudnnSetActivationDescriptor (Ptr{Void},Cint,Cint,Cdouble) desc mode CUDNN_NOT_PROPAGATE_NAN coef
    desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ActivationDesc) = desc.ptr

function activation(mode, x::CuArray{T}) where T
    h = handle()
    actdesc = ActivationDesc(mode)
    xdesc = TensorDesc(x, 4)
    y = similar(x)
    @apicall(:cudnnActivationForward,
        (Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void}),
        h, actdesc, [one(T)], xdesc, x, [zero(T)], xdesc, y)

    actdesc, xdesc, y
end

function ∇activation!(actdesc::ActivationDesc, ydesc, y::CuArray{T}, dy, x, dx) where T
    h = handle()
    @apicall(:cudnnActivationBackward,
        (Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},
        Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void}),
        h, actdesc, [one(T)], ydesc, y, ydesc, dy, ydesc, x, [one(T)], ydesc, dx)
end
