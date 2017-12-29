export getdevice, setdevice, synchronize

const CuContexts = Ptr{Void}[]

function getdevice()
    ref = Ref{Cint}()
    @apicall :cuCtxGetDevice (Ptr{Cint},) ref
    Int(ref[])
end

function setdevice(dev::Int)
    while length(CuContexts) < dev + 1
        push!(CuContexts, Ptr{Void}(0))
    end
    if CuContexts[dev+1] == Ptr{Void}(0)
        ref = Ref{Ptr{Void}}()
        @apicall :cuCtxCreate (Ptr{Ptr{Void}},Cuint,Cint) ref 0 dev
        ctx = ref[]
        CuContexts[dev+1] = ctx
        atexit(() -> @apicall :cuCtxDestroy (Ptr{Void},) ctx)

        cap = capability(dev)
        mem = round(Int, totalmem(dev) / (1024^2))
        info("device[$dev]: $(devicename(dev)), capability $(cap[1]).$(cap[2]), totalmem = $(mem) MB")
    end
    @apicall :cuCtxSetCurrent (Ptr{Void},) CuContexts[dev+1]
end

synchronize() = @apicall :cuCtxSynchronize ()
