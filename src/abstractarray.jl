export AbstractCuArray, AbstractCuVector, AbstractCuMatrx, AbstractCuVecOrMat
abstract type AbstractCuArray{T,N} end

const AbstractCuVector{T} = AbstractCuArray{T,1}
const AbstractCuMatrix{T} = AbstractCuArray{T,2}
const AbstractCuVecOrMat{T} = Union{AbstractCuVector{T},AbstractCuMatrix{T}}
