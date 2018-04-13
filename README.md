# LibCUDA.jl
[![Build Status](https://travis-ci.org/hshindo/LibCUDA.jl.svg?branch=master)](https://travis-ci.org/hshindo/LibCUDA.jl)

`LibCUDA.jl` provides a basic GPU array for [Julia](https://julialang.org/), which has been developed for deep learning library: [Merlin.jl](https://github.com/hshindo/Merlin.jl).  
For more general implementation, take a look at [JuliaGPU](https://github.com/JuliaGPU).

`LibCUDA.jl` internally loads the following libraries.
* CUBLAS
* CURAND
* NVML
* NVRTC
* CUDNN
* NCCL

## Installation
First, install `CUDA`, `CUDNN`, and `NCCL`.  
Then,
```
julia> Pkg.add("LibCUDA")
```

## Usage
```julia
T = Float32
d_x = curand(T, 10, 5)
println(Array(d_x))

fill!(d_x, 2)
println(Array(d_x))
```
