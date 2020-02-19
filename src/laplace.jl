

"""
Graph Laplacian of the line graph. With keyword `boundary = false`,
a graph Laplacian of the circle.
"""
linelaplacian(n; kwargs...) = linelaplacian(Float64,n; kwargs...)
function linelaplacian(T,n; boundary=true)
    A = SymTridiagonal(2ones(T,n), -ones(T,n-1))
    if boundary
        A[1,1] = A[end,end] = true
    end
    A
end

"""
Graph Laplacian of a `m×n` lattice.
"""
function gridlaplacian(T, m, n)
    S = sparse(T(0.0)I, n*m, n*m)
    linear = LinearIndices((1:m, 1:n))
    for i in 1:m
        for j in 1:n
            for (i2, j2) in ((i + 1, j), (i, j + 1))
                if i2 <= m && j2 <= n
                    S[linear[i, j], linear[i2, j2]] -= 1
                    S[linear[i2, j2], linear[i, j]] -= 1

                    S[linear[i, j], linear[i, j]] += 1
                    S[linear[i2, j2], linear[i2, j2]] += 1
                end
            end
        end
    end
    S
end

function gridderiv(T, m, n)
    S1 = sparse(T(0.0)*I, n*m, n*m)
    S2 = sparse(T(0.0)*I, n*m, n*m)
    S3 = sparse(T(0.0)*I, n*m, n*m)
    S4 = sparse(T(0.0)*I, n*m, n*m)

    linear = LinearIndices((1:m, 1:n))
    for i in 1:m
        for j in 1:n
            if i > 1
                S1[linear[i, j], linear[i, j]] -= 1
                S1[linear[i, j], linear[i - 1, j]] += 1
            end
            if j > 1
                S2[linear[i, j], linear[i, j]] -= 1
                S2[linear[i, j], linear[i, j - 1]] += +1
            end
            if i < m
                S3[linear[i, j], linear[i, j]] -= 1
                S3[linear[i, j], linear[i + 1, j]] += 1
            end
            if j < n
                S4[linear[i, j], linear[i, j]] -= 1
                S4[linear[i, j], linear[i, j + 1]] += +1
            end
        end
    end
    S1, S2, S3, S4
end

function boundary(A)
    m, n = size(A)
    B = zero(A)
    for i in 1:m
        for j in 1:n
            v = A[i,j]
            for (i2, j2) in ((i + 1, j), (i + 1, j +1 ), (i, j + 1))
                if 1 <= i2 <= m && 1<= j2 <= n
                    if v != A[i2, j2]
                        B[i,j] = 1.0
                        break
                    end
                end
            end
        end
    end
    B
end


function downop(T, m, n)
    S = sparse(T(0.0)I, (n÷2)*(m÷2), n*m)
    linearj = LinearIndices((1:m, 1:n))
    lineari = LinearIndices((1:m÷2, 1:n÷2))

    for i in 2:2:m
        for j in 2:2:n
            S[lineari[i÷2,j÷2], linearj[i, j]] = 1/4
            S[lineari[i÷2,j÷2], linearj[i-1, j]] = 1/4
            S[lineari[i÷2,j÷2], linearj[i, j-1]] = 1/4
            S[lineari[i÷2,j÷2], linearj[i-1, j-1]] = 1/4
        end
    end
    S
end
#=
m = n = 8
L = downop(F0, m, n); Λ = gridlaplacian(F0, m, n); Λ2 = gridlaplacian(F0, m÷2, n÷2)
heatmap([Matrix(8L*Λ*L') Matrix(Λ2) Matrix(8L*Λ*L' - Λ2)])
L = downop(F0, m, n); Λ = gridlaplacian(F0, m, n) + I/2; Λ2 = gridlaplacian(F0, m÷2, n÷2) + I
invm(x) = inv(Matrix(x))
F = Float64
L = downop(F, m, n); Λ = gridlaplacian(F, m, n)/2 + I/4; Λ2 = gridlaplacian(F, m÷2, n÷2) + I
heatmap([Matrix(L*invm(Λ)*L') Matrix(invm(Λ2)) Matrix(L*invm(Λ)*L' - invm(Λ2))])
mean(diag(L*invm(Λ)*L') - diag(invm(Λ2)))
.32007- .32008
=#
