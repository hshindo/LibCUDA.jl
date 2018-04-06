const CONTEXTS = Array{CuContext}(nthreads()*ndevices())
const FUNCTIONS = Vector{CuFunction}[]

getid() = getdevice() * nthreads() + threadid()

function getfid()
    push!(FUNCTIONS, Array{CuFunction}(length(CONTEXTS)))
    length(FUNCTIONS)
end

function getfunction!(fid::Int, kernel::String)
    id = getid()
    funs = FUNCTIONS[fid]
    if !isassigned(funs, id)
        funs[id] = CuFunction(kernel)
    end
    funs[id]
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
