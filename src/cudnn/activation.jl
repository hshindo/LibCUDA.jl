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

struct Activation
    actdesc
    xdesc
    x
    y
end

"""
* coef: floating point number to specify the clipping threashold when the activation
mode is set to CUDNN_ACTIVATION_CLIPPED_RELU or to specify the alpha coefficient
when the activation mode is set to CUDNN_ACTIVATION_ELU
"""
function activation(x::CuArray{T}, mode, coef=0.0) where T
    h = handle()
    actdesc = ActivationDesc(mode, coef)
    xdesc = TensorDesc(x, 4)
    y = similar(x)
    @apicall(:cudnnActivationForward,
        (Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void}),
        h, actdesc, [one(T)], xdesc, x, [zero(T)], xdesc, y)
    Activation(actdesc, xdesc, x, y)
end

clipped_relu(x, clip::Float64) = activation(x, CUDNN.CUDNN_ACTIVATION_CLIPPED_RELU, clip)
elu(x, alpha::Float64=1.0) = activation(x, CUDNN.CUDNN_ACTIVATION_ELU, alpha)
relu(x) = activation(x, CUDNN.CUDNN_ACTIVATION_RELU)
sigmoid(x) = activation(x, CUDNN.CUDNN_ACTIVATION_SIGMOID)
tanh(x) = activation(x, CUDNN.CUDNN_ACTIVATION_TANH)

function ∇activation!(a::Activation, dy::CuArray{T}, dx::CuArray{T}) where T
    h = handle()
    @apicall(:cudnnActivationBackward,
        (Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},
        Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void}),
        h, a.actdesc, [one(T)], a.xdesc, a.y, a.xdesc, dy, a.xdesc, a.x, [one(T)], a.xdesc, dx)
end
