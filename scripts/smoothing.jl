using Makie
using Test
using Kalman
using SuiteSparse
using BridgeSPDE
using FileIO
using Base.Iterators: flatten
using Statistics
using LinearAlgebra
using GaussianDistributions
using Trajectories
using Colors

@assert isfile(joinpath("src", "BridgeSPDE.jl"))

x = [0 0 0 0 0
     0 1 3 1 0
     0 0 0 0 0]
m, n = size(x)
mat(x) = reshape(x, (m, n))
J1 = Matrix(gridderiv(m, n)[1])
J2t = Matrix(gridderiv(m, n)[4])
image([mat(vec(x) + J2t*vec(x))  x])



sparsity(B) = nnz(B)/length(B)


downscale = 4
meteosat = [ downsample(Float64.(FileIO.load(joinpath(@__DIR__, "..","data", "meteosatf$frame.png"))), downscale) for frame in [33, 35, 40, 45]]
y = meteosat[1]
image(y, scale_plot = false)

m, n = size(y)

J1 = gridderiv(m, n)[1]
J2t = gridderiv(m, n)[4]

img(x) = image(reshape(x, (m, n)))
mat(x) = reshape(x, (m, n))

image(hcat( meteosat...))
JJ = (I + J2t)^8*(I + J1)^8
image([meteosat[1] mat(JJ*vec(meteosat[1])) meteosat[4]])
8/3

d = m*n
Λ = gridlaplacian(m, n) + I

q0 = zeros(d)
Q0 = zeros(d, d) + 2.5^2*I

dt = 0.1


m1 = quantile(flatten(meteosat), 0.17)

R = 0.1^2*I
H = I
σ = 0.3
a = σ^2*I
#@show θ1, θ2
#sparsity(B)
Δt = 1.0
T = Δt*(length(meteosat)-1)

l = 20
l2 = div(l, 2)
shape = (l2, l2, l2, l2, l2, l2)

dt = Δt / l

θ1, θ2 = 0.0, 0.0
for iter in 1:3
    global θ1, θ2
    t = T
    ν, P = copy(q0), copy(Q0)

    B = -σ^2*0.5*Λ*Λ + θ1*J1 + θ2*J2t
    Bt = typeof(B)(B')


    (ν, P), _ = Kalman.correct(Kalman.JosephForm(), Gaussian(ν, P), (Gaussian(vec(meteosat[end]), R), H))
    image(reshape(ν, (m, n)))
    μ = trajectory((t[1] => Gaussian(ν, P),))

    for segment in 3:-1:1
        for i in 2:l
            t = t - dt
            P = P .- dt*(B*P + P*Bt - a)
            ν = ν - B*ν*dt
            push!(μ, t => Gaussian(ν, P))
        end
        t = t - dt
        P = P - dt*(B*P + P*Bt - a)
        ν = ν - B*ν*dt
        (ν, P), _ = Kalman.correct(Kalman.JosephForm(), Gaussian(ν, P), (Gaussian(vec(meteosat[segment]), R), H))
        push!(μ, t => Gaussian(ν, P))
    end


    ts = μ.t
    νs = reverse(mean.(μ.x))
    Ps = reverse(cov.(μ.x))

    rs(x) = reshape(x, (m, n))
    image(hvcat(shape, rs.(νs)...))
    x0 = νs[1]
    #x = rand(Gaussian(νs[1], Symmetric(Ps[1])))
    img(νs[end-10])
    x = copy(x0); X = trajectory((ts[1]=>reshape(x, (m,n)),))
    for i in 2:length(μ)
        x = x + dt*(B*x) + σ^2*((Ps[i])\(νs[i] - x))*dt  #+ σ*sqrt(dt)*randn(d)
        t += dt
        push!(X, ts[i] => reshape(x, (m, n)))
    end


    inner(x::Vector) = dot(x,x)
    inner(x::Vector, y::Vector) = dot(x,y)
    inner(x, y) = x'*y
    inner(x) = x'*x


    μ0 = zeros(2)
    Γ = zeros(2,2)
    i = 2
    for i in 2:length(X)
        μ0 += inner([J1*vec(X.x[i-1]) J2t*vec(X.x[i-1])], vec(X.x[i]-X.x[i-1]))
        Γ += inner(inv(σ)*[J1*vec(X.x[i-1]) J2t*vec(X.x[i-1])])*dt
    end
    @show θ1, θ2 = cholesky(Symmetric(Γ)).L\μ0

    image(hvcat(shape, X.x...))

    m3 = quantile(flatten(X.x), 0.17)

    image(X.x[1])
    X.x[10]
    for i in eachindex(X.x)
        out = reshape(clamp.(X.x[i], 0, 1), (m, n))
        out2 = boundary(1.0*(out .> m3))
        #FileIO.save("trajectory$iter-$i.png", Gray.(out))
        #FileIO.save("shape$iter-$i.png", Gray.(out2))
        mi = floor.(Int, 1 + (i) ./ l)
        #@show typeof(Gray.([meteosat[mi] out out2]))
        FileIO.save(joinpath(@__DIR__, "..", "output", "all$iter-$i.png"), Gray.([meteosat[mi] out out2]))

    end
end
#run(`convert -delay 10 -resize 400% -loop 0 all{1..60}.png out.gif`)
