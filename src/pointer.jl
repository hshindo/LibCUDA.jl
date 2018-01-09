mutable struct CuPtr
    dptr::UInt64
    bytesize::Int
end

Base.convert(::Type{Ptr{T}}, p::CuPtr) where T = Ptr{T}(p.dptr)
Base.convert(::Type{UInt64}, p::CuPtr) = p.dptr
Base.unsafe_convert(::Type{Ptr{T}}, p::CuPtr) where T = Ptr{T}(p.dptr)
Base.unsafe_convert(::Type{UInt64}, p::CuPtr) = p.dptr
