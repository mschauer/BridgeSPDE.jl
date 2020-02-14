meancov(u) = u[1], u[2]
function correct(u::T, v, H; droptol=1e-7) where T
    x, Ppred = meancov(u)
    y, R = meancov(v)
    yres = y - H*x # innovation residual

    S = (H*Ppred*H' + R) # innovation covariance
    @show typeof(Ppred), typeof(S)
    K = (cholesky(Hermitian(S))\(H*Ppred))' # Kalman gain
    @show sparsity(K')
    droptol!(K', droptol)
    @show sparsity(K')
    #K = Ppred*H'*inv(S)
    #K = srsolve(S, Ppred*H', droptol=droptol)
    #sparse(SuiteSparse.CHOLMOD.spsolve(0, cholesky(outer(sprand(10,10,0.5))), SuiteSparse.CHOLMOD.Sparse(sprand(10,10,0.1))))
    x = x + K*yres
    P = (I - K*H)*Ppred*(I - K*H)' + K*R*K'
    T((x, P)), yres, S
end
