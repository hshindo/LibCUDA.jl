const CONTEXTS = Array{CuContext}(nthreads()*ndevices())
const FUNCTIONS = Vector{CuFunction}[]

getctxid() = getdevice() * nthreads() + threadid()

function getfunid!()
    push!(FUNCTIONS, Array{CuFunction}(length(CONTEXTS)))
    length(FUNCTIONS)
end

function getfun!(funid::Int, ptx::String)
    ctxid = getctxid()
    funs = FUNCTIONS[funid]
    if !isassigned(funs, ctxid)
        funs[ctxid] = CuFunction(ptx)
    end
    funs[ctxid]
end

#=
const FUNCOUNT = Ref{Int}(0)
const CUFUNCTIONS = []
function compile(kernel::String)
    ptx = NVRTC.compile(kernel)
    mod = CuModule(ptx)

    fnames = String[]
    for line in split(ptx,'\n')
        m = match(r".visible .entry (.+)\(", line) # find function name
        m == nothing && continue
        push!(fnames, String(m[1]))
    end
    length(fnames) > 1 && throw("Multiple functions are found.")
    f = CuFunction(mod, fnames[1])
    fid = FUNCOUNT[] += 1

end

function cufunction(fid::Int)
    while length(CUFUNCTIONS) < fid
        push!(CUFUNCTIONS, Array{CuFunction}(ndevices(),nthreads()))
    end
    dev = getdevice() + 1
    tid = threadid()
    CUFUNCTIONS[fid][dev,tid]
end
=#
