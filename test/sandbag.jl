workspace()
using LibCUDA
# using LibCUDA.CUDNN

setdevice(0)
x1 = curandn(Float32,4,5)
LibCUDA.relu(x1)
x2 = curand(Float32,3,5)
# y = concat(1, x1, x2)
println(x1)
y,rx = CUDNN.dropout(x1, 0.5)
println(y)
y,rx = CUDNN.dropout(x1, 0.5)
println(y)
#println(x2)

#println(y)
throw("finished")
stream0 = CuStream()
setdevice(1)
stream1 = CuStream()

x = CuArray{Float32}(10,5)
setdevice(0)
fill!(x, 3, stream=stream1)


throw("ok")
function test()
    T = Float32
    mem = LibCUDA.MemoryBuffer()
    for i = 1:10
        for j = 1:10
            x = CuArray{T}(100,10,mem=mem)
            fill!(x, 1)
            println(x)
        end
        LibCUDA.free!(mem)
    end
    LibCUDA.destroy!(mem)
end
test()
