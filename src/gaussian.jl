meancov(u) = u[1], u[2]
function correct(u::T, v, H; droptol=1e-7) where T
    x, Ppred = meancov(u)
    y, R = meancov(v)
    yres = y - H*x # innovation residual

    S = (H*Ppred*H' + R) # innovation covariance

    # not sure what is the right thing.
    if true # Dense matrix algebra
        K1 = Ppred*H'*inv(Matrix(S)) # Kalman gain
        @show sparsity(K1)
        K = droptol!(sparse(K1), droptol)
        @show sparsity(K)
    elseif false # Sparse cholesky and dense H*Ppred
        K = (cholesky(Hermitian(S))\(H*Ppred))' # Kalman gain # sparse backsolve is missing
        @show sparsity(K')
        droptol!(K', droptol)
        @show sparsity(K')
    else # Krylov method
        #K = Ppred*H'*inv(S)
        Pre = preconditioner(S, 16, 0.02)+0.01I
        K = srsolve(Pre*S, Pre*Ppred*H', tol = 5e-5, droptol=droptol)
    end
    #sparse(SuiteSparse.CHOLMOD.spsolve(0, cholesky(outer(sprand(10,10,0.5))), SuiteSparse.CHOLMOD.Sparse(sprand(10,10,0.1))))
    x = x + K*yres
    P = (I - K*H)*Ppred*(I - K*H)' + K*R*K'
    T((x, P)), yres, S
end
