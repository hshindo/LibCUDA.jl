mutable struct CuModule
    ptr::Ptr{Void}
end

function CuModule(ptx::String)
    ref = Ref{Ptr{Void}}()
    @apicall :cuModuleLoadData (Ptr{Ptr{Void}},Ptr{Void}) ref pointer(ptx)
    mod = CuModule(ref[])
    finalizer(mod, m -> @apicall :cuModuleUnload (Ptr{Void},) m)
    mod
end

Base.unsafe_convert(::Type{Ptr{Void}}, m::CuModule) = m.ptr
