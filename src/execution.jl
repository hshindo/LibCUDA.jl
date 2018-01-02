export cubox, culaunch

cubox(x) = x
cubox(x::Int) = Cint(x)
cubox{N}(t::NTuple{N,Int}) = map(Cint, t)

function cudims(n::Int)
    bx = 256
    gx = n <= bx ? 1 : ceil(Int, n/bx)
    (gx,1,1), (bx,1,1)
end

function culaunch(f::CuFunction, griddims::NTuple{3,Int}, blockdims::NTuple{3,Int}, args...; sharedmem=0, stream=C_NULL)
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
