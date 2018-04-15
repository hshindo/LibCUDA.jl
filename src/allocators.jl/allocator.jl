const Allocators = [MemoryPool() for i=1:nthreads()]

getallocator() = Allocators[1]

function setallocator(x)
    a = Allocators[1]
    free(a)
    Allocators[1] = x
end
