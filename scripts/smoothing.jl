
# Instructions
# In Julia 1.3.1 terminal run

# install and make local copy
import Pkg
Pkg.add(Pkg.PackageSpec(url="https://github.com/mschauer/BridgeSPDE.jl"))
Pkg.develop(Pkg.PackageSpec(url="https://github.com/mschauer/BridgeSPDE.jl"))

# move to package directory
cd(joinpath(Pkg.devdir(), "BridgeSPDE"))

# instantiate reproducible environment
Pkg.activate("scripts")

# run script
# include("scripts/smoothing.jl") #script is running

using Revise
using Test
using Kalman
using SparseArrays
using BridgeSPDE
using FileIO
using Base.Iterators: flatten
using Statistics
using LinearAlgebra
using GaussianDistributions
using Trajectories
using ImageTransformations
using Random
using Colors
PLOT = true
SAVE = true
if PLOT
    using Makie
end
inner(x::Vector) = dot(x,x)
inner(x::Vector, y::Vector) = dot(x,y)
inner(x, y) = x'*y
inner(x) = x'*x

@assert isfile(joinpath("src", "BridgeSPDE.jl")) # run from package directory


const F0 = Float32

θs = [(F0(0), F0(0))] 
for downscale in [8, 4, 3]
    Random.seed!(1)

    global θs
    scale = F0(8 / downscale)


    mkpath(joinpath(@__DIR__, "..", "output$downscale"))


    #ds(img, s) = imresize(img, Int(size(img,1)/downscale), Int(size(img,2)/downscale))
    ds(img, ds) = imresize(img, floor(Int,size(img,1)/ds), floor(Int,size(img,2)/ds))

    meteosat = [ ds(F0.(FileIO.load(joinpath(@__DIR__, "..","data", "meteosatf$frame.png"))), downscale) for frame in [33, 35, 40, 45]]

    if SAVE && PLOT
        for i in eachindex(meteosat)
            FileIO.save(joinpath(@__DIR__, "..", "output$downscale", "meteosat$i.png"), Gray.(meteosat[i]))

            p1 = surface(meteosat[i]; shading=false, show_axis=false, colormap = :deep)
            scale!(p1, 1.0, 1.0, 7.5)
            FileIO.save(joinpath(@__DIR__, "..", "output$downscale", "meteosurf$i.png"), p1)

        end
    end

    y = meteosat[1]

    m, n = size(y)


    Δt = F0(1.0)
    J = gridderiv(F0, m, n)
    Λ0 = gridlaplacian(F0, m, n)
    @test norm(Λ0 + (J[1]+J[2]+J[3]+J[4])) == 0

    J1 = gridderiv(F0, m, n)[1]*F0(scale)
    J2t = gridderiv(F0, m, n)[4]*F0(scale)
    img(x) = image(reshape(x, (m, n)))
    mat(x) = reshape(x, (m, n))


    d = m*n
    Λ = (Λ0 + 0.1f0 * I)*downscale

    q0 = F0(0.5) .+ zeros(F0, d)
    Q0 = zeros(F0, d, d) + F0(2.0)^2*I

    # nominally, the noise should scale with sqrt(pixel in mean)
    # the noise is not really independent, so actually it doesn't scale (so much)
    R = F0(0.05)^2*I*scale
    σ = F0(0.08*scale)
    H = I

    a = σ^2*I
    σᶥ = sparse(cholesky(Λ0+0.2I).L)

    #L = randn(5,5)
    #heatmap([inv(L*L') cov((L'\randn(5, 10000))')])

    l = floor(Int,scale)*7*[2,5,5] # not equidistant
    dt = F0(Δt / l[2])
    T = sum(l)*dt
    droptol = F0(1e-8)
    droptoli = F0(1e-8)

    iters = 1:15
    saveiters = [5, 10, 15]
    for iter in iters
        #iter = 1
        #begin
        println("\nIter $iter\n")
        θ1, θ2 = θs[end]


        t = T
        ν, P = copy(q0), copy(Q0)

        B = -σ^2/2*Λ + θ1*J1 + θ2*J2t
        B0 = -σ^2/2*Λ

        Bt = typeof(B)(B')

        @time begin
            println("backward filter")
            timecor = 0.0
            timecor += @elapsed (ν, P), _ = BridgeSPDE.correct((ν, sparse(P)), (vec(meteosat[end]), sparse(R, d,d)), H, droptol=droptoli)

            droptol!(P, droptol)


            PLOT && image(reshape(ν, (m, n)))
            μ = trajectory((t[1] => Gaussian(ν, P),))

            for segment in 3:-1:1
                for i in 2:l[segment]
                    t = t - dt
                    P = P .- dt*(B*P + P*Bt - a)
                    droptol!(P, droptol)
                    i == l[segment] && @show sparsity(P)
                    ν = ν - B*(ν*dt)
                    push!(μ, t => Gaussian(ν, P))
                end
                t = t - dt
                P = P - dt*(B*P + P*Bt - a)
                ν = ν - B*ν*dt
                #(ν, P), _ = Kalman.correct(Kalman.JosephForm(), Gaussian(ν, P), (Gaussian(vec(meteosat[segment]), R), H))
                timecor += @elapsed (ν, P), _ = BridgeSPDE.correct((ν, P), (vec(meteosat[segment]), sparse(R, d,d)), H, droptol=droptoli)
                droptol!(P, droptol)
                push!(μ, t => Gaussian(ν, P))
            end
            println("correction: $timecor")
        end

        @time begin
            println("\nforward guiding")

            #
            ts = μ.t
            νs = reverse(mean.(μ.x))
            Ps = reverse(cov.(μ.x))

            rs(x) = reshape(x, (m, n))

            x0 = νs[1]
            x = copy(x0);
            x2 = rand(Gaussian(νs[1], Symmetric(Matrix(Ps[1]))))

            global X = trajectory((ts[1]=>reshape(x, (m,n)),))
            global X2 = trajectory((ts[1]=>reshape(x2, (m,n)),))
            for i in 2:length(μ)
        #        global x,t
                x = x + dt*(B*x) + σ^2*(cholesky(Hermitian(Ps[i]) + 100droptol*I)\(νs[i] - x))*dt
                x2 = x2 + dt*(B*x2) + σ^2*(cholesky(Hermitian(Ps[i]) + 100droptol*I)\(νs[i] - x2))*dt + σ*sqrt(dt)*randn(F0, d)
                t += dt
                push!(X, ts[i] => reshape(x, (m, n)))
                push!(X2, ts[i] => reshape(x2, (m, n)))

            end
        end


        @time begin
            println("\nconjugate update")

            μ0 = zeros(F0, 2)
            Γ = zeros(F0, 2,2)
            i = 2
            for i in 2:length(X)
        #        global μ0, Γ
                x = vec(X.x[i-1])
                z = inv(a)*(vec(X.x[i]-X.x[i-1]) - B0*x*dt)
                μ0 += inner([J1*x J2t*x], z)

                Γ += inner(inv(σ)*[J1*x J2t*x])*dt # CHECK
            end
            @show θ1, θ2 = cholesky(Symmetric(Γ))\μ0
            push!(θs, (θ1, θ2))
        end

        @time begin
            println("\ncreate images")

            m3 = quantile(flatten(X.x), 0.17)

            PLOT && image(X.x[1])
            if SAVE && iter in saveiters
                for i in eachindex(X.x)[1:5:end]
                    global out = reshape(clamp.(X.x[i], 0, 1), (m, n))

                    out2 = boundary(F0.(out .> m3))
                    out3 = reshape(clamp.(X2.x[i], 0, 1), (m, n))
                    mi = findmin(map(k->abs(k - i), cumsum(l[2:end]).-l[1]))[2] # which image is closest
                    FileIO.save(joinpath(@__DIR__, "..", "output$downscale", "img$iter-$i.png"), Gray.(out))

                    if PLOT
                        p1 = surface(out; shading=false, show_axis=false, colormap = :deep)
                        scale!(p1, 1.0, 1.0, 7.5)
                        FileIO.save(joinpath(@__DIR__, "..", "output$downscale", "surf$iter-$i.png"), p1)
                        p2 = surface(out3; shading=false, show_axis=false, colormap = :deep)
                        scale!(p2, 1.0, 1.0, 7.5)
                        FileIO.save(joinpath(@__DIR__, "..", "output$downscale", "surfr$iter-$i.png"), p2)

                    end
                end
                run(`ffmpeg -y -r 40 -f image2 -i output$downscale/img$iter-%d.png -vcodec libx264 -crf 25 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p img$downscale-$iter.mp4`)
                PLOT && run(`ffmpeg -y -r 40 -f image2 -i output$downscale/surf$iter-%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p surf$downscale-$iter.mp4`)
                PLOT && run(`ffmpeg -y -r 40 -f image2 -i output$downscale/surfr$iter-%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p surfr$downscale-$iter.mp4`)

            end
        end
    end

    println("Done downscale= $downscale.")
end

# save thetas in scale of pixels of final resolution
downscale = 3
scale = F0(8 / downscale)
Δt = F0(1.0)

l = floor(Int,scale)*7*[2,5,5] # not equidistant
dt = F0(Δt / l[2])
T = sum(l)*dt

writedlm("thetas.txt", [first.(θs)*scale*T last.(θs)*scale*T])

lines(first.(θs)*scale*T);
lines!(last.(θs)*scale*T)
