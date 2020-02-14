using IterativeSolvers
using SparseArrays
using Random
using Printf

#########################
# Method Implementation #
#########################

@inline function omega(t, s)
    angle = sqrt(2.)/2
    ns = norm(s)
    nt = norm(t)
    ts = dot(t,s)
    rho = abs(ts/(nt*ns))
    om = ts/(nt*nt)
    if rho < angle
        om = om*convert(typeof(om),angle)/rho
    end
    om
end

#s = 8, , ,
function sidrs!(X0, A, C::T; s::Number = 4, tol=10sqrt(eps(real(eltype(C)))), droptol=(eps(real(eltype(C)))), maxiter=size(A, 2), verbose::Bool=false) where {T}
	verbose && @show tol
	X = X0

    verbose && @printf("=== idrs ===\n%4s\t%7s\n","iter","resnorm")
    R = C - A(X)
    normR = norm(R)
	iter = 1

    if normR <= tol # Initial guess is a good enough solution
        return X
    end

    Z = zero(C)
	d = size(C,1)
    P = T[Diagonal(rand(d) .- 0.5) + sprandn(d,d, 1/d) for k in 1:s]
    U = T[copy(C) for k in 1:s]
    G = T[copy(C) for k in 1:s] #
    Q = copy(Z)
    V = copy(Z)

    M = Matrix{eltype(C)}(I,s,s)
    f = zeros(eltype(C),s)
    c = zeros(eltype(C),s)

    om::eltype(C) = 1
    while normR > tol && iter ≤ maxiter
        for i in 1:s
            f[i] = dot(P[i], R)
        end
        for k in 1:s

            # Solve small system and make v orthogonal to P

            c = LowerTriangular(M[k:s,k:s])\f[k:s]
            V = c[1] * G[k]
            Q = c[1] * U[k]

            for i = k+1:s
                V += c[i-k+1] * G[i]
                Q += c[i-k+1] * U[i]
            end

            # Compute new U[:,k] and G[:,k], G[:,k] is in space G_j
            V = R - V

            U[k] = Q + om * V
            G[k] = A(U[k])

            # Bi-orthogonalise the new basis vectors

            for i in 1:k-1
                alpha = dot(P[i],G[k])/M[i,i]
                G[k] -= alpha * G[i]
                U[k] -= alpha * U[i]
            end



            # New column of M = P'*G  (first k-1 entries are zero)

            for i in k:s
                M[i,k] = dot(P[i],G[k])
            end

            #  Make r orthogonal to q_i, i = 1..k

            beta = f[k]/M[k,k]
            R -= beta * G[k]
            X += beta * U[k]

			droptol!(X, droptol)

            normR = norm(R)
            verbose && @printf("%3d\t%1.2e\t%1.2e\n",iter,normR,sparsity(X))
            if normR < tol || iter == maxiter
                iter == maxiter && warn("not converged")
                break
            end
            if k < s
                f[k+1:s] .-=  beta*M[k+1:s,k]
            end
            iter += 1
        end # k in 1:s

		#@show sparsity(R), sparsity(X), sparsity(Q)

        # Now we have sufficient vectors in G_j to compute residual in G_j+1
        # Note: r is already perpendicular to P so v = r
        copyto!(V, R)
        Q = A(V)
        om = omega(Q, R)
        R -= om .* Q
        X += om .* V
		droptol!(X, droptol)

        normR = norm(R)

        iter += 1
    end
    verbose && print("\n")
    X
end

slyap_op(B) = X -> B*X + X*B'
linl_op(B) = X -> B*X
linr_op(B) = X -> X*B
slyapunov(B, C; droptol = 1e-7) = sidrs!(copy(C), slyap_op(B), -C; maxiter=prod(size(B)), verbose=true, droptol = droptol)
slsolve(B, Y; droptol = 1e-7) = sidrs!(copy(Y), linl_op(B), Y; maxiter=prod(size(B)), verbose=true, droptol = droptol)
srsolve(B, Y; droptol = 1e-7) = sidrs!(copy(Y), linr_op(B), Y; maxiter=prod(size(B)), verbose=true, droptol = droptol)


"""
Compute lyaponov equation using the Kronecker product for testing.
"""
kronlyap(B,C) = -reshape(inv(kron(B, Diagonal(I,size(B,1))) + kron(Diagonal(I,size(B,1)), B))*vec(C), size(C))
