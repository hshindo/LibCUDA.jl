mutable struct DropoutDesc
    ptr::Cptr
    states

    function DropoutDesc(droprate::Float64, seed::Int=0)
        ref = Ref{Cptr}()
        @cudnn :cudnnCreateDropoutDescriptor (Ptr{Cptr},) ref
        ptr = ref[]

        h = gethandle()
        ref = Ref{Csize_t}()
        @cudnn :cudnnDropoutGetStatesSize (Cptr,Ptr{Csize_t}) h ref
        states = CuArray{UInt8}(Int(ref[]))
        desc = new(ptr, states)

        @cudnn(:cudnnSetDropoutDescriptor,
            (Cptr,Cptr,Cfloat,Cptr,Csize_t,Culonglong),
            desc, h, droprate, states, length(states), seed)
        finalizer(desc, x -> @cudnn :cudnnDestroyDropoutDescriptor (Cptr,) x.ptr)
        
        desc
    end
end

Base.unsafe_convert(::Type{Cptr}, desc::DropoutDesc) = desc.ptr

function dropout(x::CuArray{T,N}, droprate::Float64) where {T,N}
    dropdesc = DropoutDesc(Val{droprate})
    xdesc = TensorDesc(x, 4)

    ref = Ref{Csize_t}()
    @cudnn :cudnnDropoutGetReserveSpaceSize (Cptr,Ptr{Csize_t}) xdesc ref
    reservespace = CuArray{UInt8}(Int(ref[]))

    h = gethandle()
    y = similar(x)
    @cudnn(:cudnnDropoutForward,
        (Cptr,Cptr,Cptr,Cptr,
        Cptr,Cptr,Cptr,Csize_t),
        h, dropdesc, xdesc, x, xdesc, y, reservespace, length(reservespace))

    y, reservespace
end

function ∇dropout!(dy, dx, droprate, reservespace)
    dropdesc = DropoutDesc(Val{droprate})
    xdesc = TensorDesc(dy, 4)
    h = gethandle()
    @cudnn(:cudnnDropoutBackward,
        (Cptr,Cptr,Cptr,Cptr,
        Cptr,Cptr,Cptr,Csize_t),
        h, dropdesc, xdesc, dy, xdesc, dx, reservespace, length(reservespace))
end
