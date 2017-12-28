export getdevice, setdevice, synchronize

function getdevice()
    ref = Ref{Cint}()
    @apicall :cuCtxGetDevice (Ptr{Cint},) ref
    Int(ref[])
end

function setdevice(dev::Int)
    @apicall :cuCtxSetCurrent (Ptr{Void},) CuContexts[dev+1]
end

synchronize() = @apicall :cuCtxSynchronize ()
