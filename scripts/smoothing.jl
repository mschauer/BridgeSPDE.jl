#using Makie
using Revise
using Test
using Kalman
using SparseArrays
#using SuiteSparse
using BridgeSPDE
using FileIO
using Base.Iterators: flatten
using Statistics
using LinearAlgebra
using GaussianDistributions
using Trajectories
using Colors
PLOT = false
SAVE = false

inner(x::Vector) = dot(x,x)
inner(x::Vector, y::Vector) = dot(x,y)
inner(x, y) = x'*y
inner(x) = x'*x

@assert isfile(joinpath("src", "BridgeSPDE.jl"))





const F0 = Float32

downscale = 2 # was 4
meteosat = [ downsample(F0.(FileIO.load(joinpath(@__DIR__, "..","data", "meteosatf$frame.png"))), downscale) for frame in [33, 35, 40, 45]]
y = meteosat[1]
PLOT && image(y, scale_plot = false)

m, n = size(y)


Δt = F0(1.0)
T = Δt*(length(meteosat)-1)


J1 = gridderiv(F0, m, n)[1]
J2t = gridderiv(F0, m, n)[4]

img(x) = image(reshape(x, (m, n)))
mat(x) = reshape(x, (m, n))

PLOT && image(hcat( meteosat...))

θ1, θ2 = [2.5, 5.4]*2/downscale # left, down
#
shi = vec(meteosat[1])
shi = (I + J1)^ceil(Int, θ1*T)*shi
shi = (I + J2t)^ceil(Int, θ2*T)*shi
if PLOT
    p = image([meteosat[1].*(1 .- boundary(mat(meteosat[1].>0.3)))
       meteosat[end].*(1 .- boundary(mat(shi.>0.3)))])
    #save("shift.png", p)
end

d = m*n
Λ = gridlaplacian(F0, m, n) + 0.1*I

q0 = zeros(F0, d)
Q0 = zeros(F0, d, d) + F0(2.5)^2*I



#m1 = quantile(flatten(meteosat), 0.17)

R = F0(0.1)^2*I
H = I
σ = F0(0.3)
a = σ^2*I



l = 40
l2 = div(l, 2)
shape = (l2, l2, l2, l2, l2, l2)

dt = F0(Δt / l)
droptol = F0(1e-8)
droptoli = F0(1e-8)

θ1, θ2 = F0(0.0), F0(0.0)
for iter in 1:10
    global θ1, θ2
    t = T
    ν, P = copy(q0), copy(Q0)

    #B = -σ^2*0.5*Λ*Λ + θ1*J1 + θ2*J2t
    B = -σ^2/2*Λ + θ1*J1 + θ2*J2t
    B0 = -σ^2/2*Λ

    Bt = typeof(B)(B')


#    @time (ν_, P_), _ = Kalman.correct(Kalman.JosephForm(), Gaussian(ν, P), (Gaussian(vec(meteosat[end]), R), H))
    @time (ν, P), _ = BridgeSPDE.correct((ν, sparse(P)), (vec(meteosat[end]), sparse(R, d,d)), H, droptol=droptoli)

    droptol!(P, droptol)
#    @test norm(P - P_) < 1e-7
#    @test norm(ν - ν_) < 1e-7


    PLOT && image(reshape(ν, (m, n)))
    μ = trajectory((t[1] => Gaussian(ν, P),))

    for segment in 3:-1:1
        for i in 2:l
            t = t - dt
            P = P .- dt*(B*P + P*Bt - a)
            droptol!(P, droptol)
            i == l && @show sparsity(P)
            ν = ν - B*(ν*dt)
            push!(μ, t => Gaussian(ν, P))
        end
        t = t - dt
        P = P - dt*(B*P + P*Bt - a)
        ν = ν - B*ν*dt
        #(ν, P), _ = Kalman.correct(Kalman.JosephForm(), Gaussian(ν, P), (Gaussian(vec(meteosat[segment]), R), H))
        (ν, P), _ = BridgeSPDE.correct((ν, P), (vec(meteosat[segment]), sparse(R, d,d)), H, droptol=droptoli)
        droptol!(P, droptol)
        push!(μ, t => Gaussian(ν, P))
    end


    ts = μ.t
    νs = reverse(mean.(μ.x))
    global Ps = reverse(cov.(μ.x))

    rs(x) = reshape(x, (m, n))
    PLOT && image(hvcat(shape, rs.(νs)...))
    x0 = νs[1]
    #x0 = rand(Gaussian(νs[1], Symmetric(Matrix(Ps[1]))))
    PLOT && img(νs[end-10])
    x = copy(x0);
    X = trajectory((ts[1]=>reshape(x, (m,n)),))
    for i in 2:length(μ)
        x = x + dt*(B*x) + σ^2*(cholesky(Hermitian(Ps[i]))\(νs[i] - x))*dt # + σ*sqrt(dt)*randn(d)
        t += dt
        push!(X, ts[i] => reshape(x, (m, n)))
    end





    μ0 = zeros(F0, 2)
    Γ = zeros(F0, 2,2)
    i = 2
    for i in 2:length(X)
        x = vec(X.x[i-1])
        z = inv(a)*(vec(X.x[i]-X.x[i-1]) - B0*x*dt)
        μ0 += inner([J1*x J2t*x], z)

        Γ += inner(inv(σ)*[J1*x J2t*x])*dt # CHECK
    end
    @show θ1, θ2 = cholesky(Symmetric(Γ))\μ0

    PLOT && image(hvcat(shape, X.x...))

    m3 = quantile(flatten(X.x), 0.17)

    PLOT && image(X.x[1])
    if SAVE
        for i in eachindex(X.x)
            out = reshape(clamp.(X.x[i], 0, 1), (m, n))
            out2 = boundary(F0.(out .> m3))
            #FileIO.save("trajectory$iter-$i.png", Gray.(out))
            #FileIO.save("shape$iter-$i.png", Gray.(out2))
            mi = floor.(Int, 1 + (i) ./ l)
            #@show typeof(Gray.([meteosat[mi] out out2]))
            FileIO.save(joinpath(@__DIR__, "..", "output", "all$iter-$i.png"), Gray.([meteosat[mi] out out2]))

        end
    end
end
#run(`convert -delay 10 -resize 400% -loop 0 all{1..60}.png out.gif`)
#  convert -delay 10 -resize 400% -loop 0 output/all3-{1..60}.png out.gif
