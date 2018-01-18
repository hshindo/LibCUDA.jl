workspace()
using LibCUDA

LibCUDA.testcu()

T = Float32
x = curandn(T, 5, 9)
CUDNN.batch_rnn(x, [4,3,2])
y = CUDNN.catvec(xs)
display(y)
throw("finish")
#=
T = Float32
xs = [curandn(T,5,4) for i=1:2]
display(xs[1])
println()
display(xs[2])
println()
#display(xs[3])
#println()
y = cat(2, xs...)
display(y)
throw("finish")
=#

function bench()
    T = Float32
    y = CuArray{T}(50, 50, 50)
    x = CuArray{T}(50, 50, 50)
    #suby = view(y, 10:59, 10:59, 10:59)
    @time begin
        for i = 1:20
            cat(1, x, y)
        end
        synchronize()
    end
end
# bench()
