

downsample(x, s = 10) = [mean(x[i:i+s-1,j:j+s-1]) for i in 1:s:size(x,1)-s, j in 1:s:size(x, 2)-s]


function gridlaplacian(m, n)
    S = sparse(0.0I, n*m, n*m)
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

function gridderiv(m, n)
    S1 = sparse(0.0I, n*m, n*m)
    S2 = sparse(0.0I, n*m, n*m)
    S3 = sparse(0.0I, n*m, n*m)
    S4 = sparse(0.0I, n*m, n*m)

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
    B = zeros(A)
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
