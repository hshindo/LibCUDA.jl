mutable struct ActivationDesc
    ptr::Ptr{Void}
end

function ActivationDesc(mode, coef::Float64)
    ref = Ref{Ptr{Void}}()
    @apicall :cudnnCreateActivationDescriptor (Ptr{Ptr{Void}},) ref
    desc = ActivationDesc(ref[])
    finalizer(desc, x -> @apicall :cudnnDestroyActivationDescriptor (Ptr{Void},) x)

    @apicall :cudnnSetActivationDescriptor (Ptr{Void},Cint,Cint,Cdouble) desc mode CUDNN_NOT_PROPAGATE_NAN coef
    desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ActivationDesc) = desc.ptr

"""
* coef: floating point number to specify the clipping threashold when the activation
mode is set to CUDNN_ACTIVATION_CLIPPED_RELU or to specify the alpha coefficient
when the activation mode is set to CUDNN_ACTIVATION_ELU
"""
function activation(x::CuArray{T}, mode, coef=0.0) where T
    actdesc = ActivationDesc(mode, coef)
    xdesc = TensorDesc(x, 4)
    y = similar(x)
    @apicall(:cudnnActivationForward,
        (Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void}),
        handle(), actdesc, T[1], xdesc, x, T[0], xdesc, y)
    y, (actdesc,xdesc)
end

function ∇activation!(y::CuArray{T}, dy, x, dx, actdesc, xdesc) where T
    @apicall(:cudnnActivationBackward,
        (Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},
        Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void}),
        handle(), actdesc, T[1], xdesc, y, xdesc, dy, xdesc, x, T[1], xdesc, dx)
end
