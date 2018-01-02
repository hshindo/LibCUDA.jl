using LibCUDA

setdevice(0)
function test()
    T = Float32
    p = Var(Cint[1,2,3])
    cup = Var(cu(p.data))
    x = Var(randn(T,5,3))
    cux = Var(cu(x.data))
    y = softmax_crossentropy(p, x)
    println(y)
    cuy = softmax_crossentropy(cup, cux)
    println(cuy)
end
test()
