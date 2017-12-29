mutable struct DropoutDesc
    ptr::Ptr{Void}
end

function DropoutDesc(droprate::Float64; seed=rand(UInt64))
    ref = Ref{Ptr{Void}}()
    @apicall :cudnnCreateDropoutDescriptor (Ptr{Ptr{Void}},) ref
    desc = DropoutDesc(ref[])
    finalizer(desc, x -> @apicall :cudnnDestroyDropoutDescriptor (Ptr{Void},) x)

    h = handle()
    ref = Ref{Csize_t}()
    @apicall :cudnnDropoutGetStatesSize (Ptr{Void},Ptr{Csize_t}) h ref
    states = CuArray{UInt8}(Int(ref[]))

    @apicall(:cudnnSetDropoutDescriptor,
        (Ptr{Void},Ptr{Void},Cfloat,Ptr{Void},Csize_t,Culonglong),
        desc, h, droprate, states, length(states), seed)
    desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::DropoutDesc) = desc.ptr

function dropout(x::CuArray, droprate::Float64; seed=rand(UInt64))
    dropdesc = Dropout(droprate, seed)
    xdesc = TensorDesc(x, 4)

    ref = Ref{Csize_t}()
    @apicall :cudnnDropoutGetReserveSpaceSize (Ptr{Void},Ptr{Csize_t}) xdesc ref
    reservespace = CuArray{UInt8}(Int(ref[]))

    @apicall(:cudnnDropoutForward,
        (Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},
        Ptr{Void},Ptr{Void},Ptr{Void},Csize_t),
        handle(), dropdesc, xdesc, x, ydesc, y, reservespace, length(reservespace))

    dropdesc, xdesc, x, ydesc, y, reservespace
end

function ∇dropout!(dropdesc::DropoutDesc, dydesc, dy, dxdesc, dx, reservespace)
    @apicall(:cudnnDropoutBackward,
        (Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},
        Ptr{Void},Ptr{Void},Ptr{Void},Csize_t),
        handle(), dropdesc, dydesc, dy, dxdesc, dx, reservespace, length(reservespace))
end
