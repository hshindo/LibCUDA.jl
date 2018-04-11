export CuLinearArray
const CuLinearArray{T,N} = Union{CuArray{T,N},CuSubArray{T,N,true}}

function Base.copy!(dest::Array{T}, src::CuLinearArray{T}, n=length(src)) where T
    nbytes = n * sizeof(T)
    @apicall :cuMemcpyDtoH (Ptr{Void},Ptr{Void},Csize_t) dest src nbytes
    dest
end
function Base.copy!(dest::CuLinearArray{T}, src::Array{T}, n=length(src); stream=C_NULL) where T
    nbytes = n * sizeof(T)
    @apicall :cuMemcpyHtoDAsync (Ptr{Void},Ptr{Void},Csize_t,Ptr{Void}) dest src nbytes stream
    dest
end
function Base.copy!(dest::CuLinearArray{T}, src::CuLinearArray{T}, n=length(src); stream=C_NULL) where T
    nbytes = n * sizeof(T)
    @apicall :cuMemcpyDtoDAsync (Ptr{Void},Ptr{Void},Csize_t,Ptr{Void}) dest src nbytes stream
    dest
end
function Base.copy!(dest::CuLinearArray{T}, doffs::Int, src::CuLinearArray{T}, soffs::Int, n::Int; stream=C_NULL) where T
    p_dest = pointer(dest, doffs)
    p_src = pointer(src, soffs)
    nbytes = n * sizeof(T)
    @apicall :cuMemcpyDtoDAsync (Ptr{Void},Ptr{Void},Csize_t,Ptr{Void}) p_dest p_src nbytes stream
    dest
end
