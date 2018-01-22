export geam!

for (f,T,Ct) in (
    (:(:cublasSgeam),:Float32,:Cfloat),
    (:(:cublasDgeam),:Float64,:Cdouble))
    @eval begin
        function geam!(tA::Char, tB::Char, alpha::$T, A::CuMatrix{$T},
            beta::$T, B::CuMatrix{$T}, C::CuMatrix{$T})

            m = size(A, tA == 'N' ? 1 : 2)
            n = size(A, tA == 'N' ? 2 : 1)
            if m != size(C,1) || n != size(C,2)
                throw(DimensionMismatch())
            end

            @cublas($f, (
                Ptr{Void},Cint,Cint,Cint,Cint,
                Ptr{$Ct},Ptr{$Ct},Cint,
                Ptr{$Ct},Ptr{$Ct},Cint,
                Ptr{$Ct},Cint),
                handle(), cublasop(tA), cublasop(tB), m, n,
                [alpha], A, stride(A,2),
                [beta], B, stride(B,2),
                C, stride(C,2))
            C
        end
    end
end

function Base.transpose(x::CuMatrix{T}) where T
    t = similar(x, size(x,2), size(x,1))
    geam!('T', 'N', T(1), x, T(0), t, t)
    t
end
Base.transpose(x::CuVector) = transpose(reshape(x,length(x),1))
