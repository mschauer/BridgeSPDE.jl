sparsity(B) = Float32(nnz(B)/length(B))
sparsity(B::Matrix) = 1.0f0
sparsity(B::Adjoint) = sparsity(B.parent)

function preconditioner(B, s, tol=1e-2)
    upsample(droptol!(sparse(inv(downsample2(B, s)))/s/s, tol), s)
end

#downsample(x, s = 10) = [mean(x[i:i+s-1,j:j+s-1]) for i in 1:s:size(x,1)-s, j in 1:s:size(x, 2)-s]
downsample(x, s = 10) = [mean(x[i:i+s-1,j:j+s-1]) for i in 1:s:size(x,1)-s+1, j in 1:s:size(x, 2)-s+1]
upsample(x, s = 10) = repeat(x, inner=(s, s))
