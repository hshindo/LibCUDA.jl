using Base.Test
using LibCUDA

Base.isapprox(x::Array, y::CuArray) = isapprox(x, Array(y))

setdevice(0)
T = Float32

@testset "array" for i=1:5
    x = randn(T, 10, 5)
    cux = CuArray(x)
    fill!(cux, 3)
    @test all(v -> v == T(3), Array(cux))
end

@testset "reduce" for i = 1:5
    x = randn(T, 10, 5, 3)
    for dim = 1:ndims(x)
        y = sum(x, dim)
        cuy = sum(CuArray(x), dim)
        @test y ≈ cuy

        y = mean(x, dim)
        cuy = mean(CuArray(x), dim)
        @test y ≈ cuy

        y, idx = findmax(x, dim)
        cuy, cuidx = findmax(CuArray(x), dim)
        @test y ≈ cuy
        # @test idx ≈ cuidx

        y, idx = findmin(x, dim)
        cuy, cuidx = findmin(CuArray(x), dim)
        @test y ≈ cuy

        y = maximum(abs, x, dim)
        cuy = maximum(abs, CuArray(x), dim)
        @test y ≈ cuy
    end
end

@testset "BLAS" for i = 1:5
    x = randn(T, 10, 5)
    y = randn(T, 10, 5)
    cuy = CuArray(y)
    BLAS.axpy!(T(1), x, y)
    BLAS.axpy!(T(1), CuArray(x), cuy)
    @test y ≈ Array(cuy)

    A = randn(T, 10, 5)
    x = randn(T, 5)
    y = randn(T, 10)
    cuy = CuArray(y)
    BLAS.gemv!('N', T(1), A, x, T(0), y)
    BLAS.gemv!('N', T(1), CuArray(A), CuArray(x), T(0), cuy)
    @test y ≈ Array(cuy)

    A = randn(T, 10, 5)
    B = randn(T, 5, 4)
    C = randn(T, 10, 4)
    cuC = CuArray(C)
    BLAS.gemm!('N', 'N', T(1), A, B, T(0), C)
    BLAS.gemm!('N', 'N', T(1), CuArray(A), CuArray(B), T(0), cuC)
    @test C ≈ Array(cuC)

    As = [randn(T,10,5) for i=1:10]
    Bs = [randn(T,10,5) for i=1:10]
    cuAs = map(CuArray, As)
    cuBs = map(CuArray, Bs)
    Cs = [BLAS.gemm('N','T',T(1),As[i],Bs[i]) for i=1:length(As)]
    cuCs = CUBLAS.gemm_batched('N', 'T', T(1), cuAs, cuBs)
    for (C,cuC) in zip(Cs,cuCs)
        @test C ≈ Array(cuC)
    end
end
