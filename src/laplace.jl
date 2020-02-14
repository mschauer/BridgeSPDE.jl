

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
