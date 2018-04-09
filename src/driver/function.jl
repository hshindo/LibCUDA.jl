export CuFunction

mutable struct CuFunction
    ptr::Ptr{Void}
    mod::CuModule # avoid CuModule gc-ed
end

function CuFunction(mod::CuModule, name::String)
    ref = Ref{Ptr{Void}}()
    @apicall :cuModuleGetFunction (Ptr{Ptr{Void}},Ptr{Void},Cstring) ref mod name
    CuFunction(ref[], mod)
end

function CuFunction(ptx::String)
    mod = CuModule(ptx)
    fnames = String[]
    for line in split(ptx,'\n')
        m = match(r".visible .entry (.+)\(", line) # find function name
        m == nothing && continue
        push!(fnames, String(m[1]))
    end
    length(fnames) > 1 && throw("Multiple functions are found.")
    CuFunction(mod, fnames[1])
end

Base.unsafe_convert(::Type{Ptr{Void}}, f::CuFunction) = f.ptr

function (f::CuFunction)(griddims::NTuple{3,Int}, blockdims::NTuple{3,Int})
    throw("Not implemented yet.")
end

function culaunch2(f::CuFunction, griddims::NTuple{3,Int}, blockdims::NTuple{3,Int}, args...; sharedmem=0, stream=C_NULL)
    argptrs = Ptr{Void}[pointer_from_objref(cubox(a)) for a in args]
    @apicall(:cuLaunchKernel, (
        Ptr{Void},           # function
        Cuint,Cuint,Cuint,      # grid dimensions (x, y, z)
        Cuint,Cuint,Cuint,      # block dimensions (x, y, z)
        Cuint,                  # shared memory bytes,
        Ptr{Void},             # stream
        Ptr{Ptr{Void}},         # kernel parameters
        Ptr{Ptr{Void}}),         # extra parameters
        f,
        griddims[1], griddims[2], griddims[3],
        blockdims[1], blockdims[2], blockdims[3],
        sharedmem, stream, argptrs, C_NULL)
end
