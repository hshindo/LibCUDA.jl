workspace()
using LibCUDA

function bench()
    T = Float32
    x = curand(T,2048,2048)
    streams = [CuStream(LibCUDA.CU_STREAM_NON_BLOCKING) for i=1:10]
    handles = [CUDNN.Handle() for i=1:10]
    for i = 1:10
        CUDNN.setstream(handles[i], streams[i])
    end
    for i = 1:100
        for j = 1:10
            y = CUDNN.sum(x,2,handles[1])
        end
    end
    synchronize()
end
#sum(Array(x),2)
@time bench()
