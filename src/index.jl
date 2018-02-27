import Base: getindex

function getindex(x::CuArray{T}, inds...) where T
    subx = view(x, inds...)
    y = CuArray{T}(size(subx))
    copy!(y, subx)
    y
end
