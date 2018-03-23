function gethandle(dev::Int)
    ref = Ref{Ptr{Void}}()
    @apicall :nvmlDeviceGetHandleByIndex (Cuint,Ptr{Void}) dev ref
    ref[]
end

"""
Get information about processes with a compute context on a device
"""
function running_processes(dev::Int)
    h = gethandle(dev)
    ref_count = Ref{Ptr{Cuint}}()
    ref_infos = Ref{Ptr{Void}}()
    @apicall :nvmlDeviceGetComputeRunningProcesses (Ptr{Void},Ptr{Cuint},Ptr{Void}) h ref_count ref_infos
end

function free_devices(maxcount::Int)
    devs = Int[]
    for i = 0:ndevices()-1
        length(devs) >= maxcount && break
        # setdevice(i) do
        h = gethandle(i)
        res = @apicall_nocheck :nvmlDeviceGetComputeRunningProcesses (Ptr{Void},Ptr{Cuint},Ptr{Void}) h Ref{Cuint}(0) C_NULL
        res == 0 && push!(devs,i)
    end
    devs
end