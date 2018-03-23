function UtilizationRates()
    @apicall :nvmlDeviceGetUtilizationRates (Ptr{Cint},Cint,Cint) ref code dev
end
