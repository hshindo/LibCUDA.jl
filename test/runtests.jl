using Base.Test
using LibCUDA

Base.isapprox(x::Array, y::CuArray) = isapprox(x, Array(y))

setdevice(0)
T = Float32

@testset "reduce" for i = 1:5
    x = randn(T, 10, 5, 3)
    for dim = 1:ndims(x)
        y = sum(x, dim)
        cuy = sum(cu(x), dim)
        @test y ≈ cuy

        y = mean(x, dim)
        cuy = mean(cu(x), dim)
        @test y ≈ cuy

        y, idx = findmax(x, dim)
        cuy, cuidx = findmax(cu(x), dim)
        @test y ≈ cuy
        @test idx ≈ cuidx
    end
end

@testset "BLAS" for i = 1:5
    x = randn(T, 10, 5)
    y = randn(T, 10, 5)
    cuy = cu(y)
    BLAS.axpy!(T(1), x, y)
    BLAS.axpy!(T(1), cu(x), cuy)
    @test y ≈ Array(cuy)

    A = randn(T, 10, 5)
    x = randn(T, 5)
    y = randn(T, 10)
    cuy = cu(y)
    BLAS.gemv!('N', T(1), A, x, T(0), y)
    BLAS.gemv!('N', T(1), cu(A), cu(x), T(0), cuy)
    @test y ≈ Array(cuy)

    A = randn(T, 10, 5)
    B = randn(T, 5, 4)
    C = randn(T, 10, 4)
    cuC = cu(C)
    BLAS.gemm!('N', 'N', T(1), A, B, T(0), C)
    BLAS.gemm!('N', 'N', T(1), cu(A), cu(B), T(0), cuC)
    @test C ≈ Array(cuC)
end
