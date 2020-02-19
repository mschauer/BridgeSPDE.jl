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

@assert isfile(joinpath("src", "BridgeSPDE.jl"))
Random.seed!(1)



const F0 = Float32

#downscale = 4 # was 4, can be 2, 4, 8
#scale = 8 ÷ downscale
downscale = 3 # was 4, can be 3, 4, 8
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
PLOT && image(y, scale_plot = false)

m, n = size(y)


Δt = F0(1.0)
T = Δt*(length(meteosat)-1)


J1 = gridderiv(F0, m, n)[1]*F0(scale)
J2t = gridderiv(F0, m, n)[4]*F0(scale)

img(x) = image(reshape(x, (m, n)))
mat(x) = reshape(x, (m, n))

PLOT && image(hcat( meteosat...))

#=
θ1, θ2 = Float32[0.080375865, 0.7903045] # left, down
shi = vec(meteosat[1])
shi = (I + J1/F0(scale))^ceil(Int, θ1*F0(scale)*T)*shi
shi = (I + J2t/F0(scale))^ceil(Int, θ2*F0(scale)*T)*shi
if true #PLOT
    p = image([meteosat[1].*(1 .- boundary(mat(meteosat[1].>0.3)))
       meteosat[end].*(1 .- boundary(mat(shi.>0.3)))])
    save("shift.png", p)
end
=#

d = m*n
Λ = (gridlaplacian(F0, m, n) + 0.1f0 * I)*downscale

q0 = F0(0.5) .+ zeros(F0, d)
Q0 = zeros(F0, d, d) + F0(2.0)^2*I

# nominally, the noise should scale with sqrt(pixel in mean)
# the noise is not really independent, so actually it doesn't scale (so much)
R = F0(0.05)^2*scale*I
σ = F0(0.1*scale)
H = I

a = σ^2*I



l = scale*7*[2,5,5] # not equidistant
dt = F0(Δt / l[2])
T = sum(l)*dt
droptol = F0(1e-8)
droptoli = F0(1e-8)

iters = 1:10
saveiters = [1, 5, 10, 15]
#θs = [(F0(0), F0(0))] # 2.4448118, 8.101286
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
    #    @time (ν_, P_), _ = Kalman.correct(Kalman.JosephForm(), Gaussian(ν, P), (Gaussian(vec(meteosat[end]), R), H))
        timecor += @elapsed (ν, P), _ = BridgeSPDE.correct((ν, sparse(P)), (vec(meteosat[end]), sparse(R, d,d)), H, droptol=droptoli)

        droptol!(P, droptol)
    #    @test norm(P - P_) < 1e-7
    #    @test norm(ν - ν_) < 1e-7


        PLOT && image(reshape(ν, (m, n)))
        μ = trajectory((t[1] => Gaussian(ν, P),))

        for segment in 3:-1:1
    #        global t, P, ν, timecor
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
    #PLOT && image(hvcat(shape, X.x...))

        m3 = quantile(flatten(X.x), 0.17)

        PLOT && image(X.x[1])
        if SAVE && iter in saveiters
            for i in eachindex(X.x)
                global out = reshape(clamp.(X.x[i], 0, 1), (m, n))

                out2 = boundary(F0.(out .> m3))
                out3 = reshape(clamp.(X2.x[i], 0, 1), (m, n))
                #FileIO.save("trajectory$iter-$i.png", Gray.(out))
                #FileIO.save("shape$iter-$i.png", Gray.(out2))
                #mi = floor.(Int, 1 + (i) ./ l)
                mi = findmin(map(k->abs(k - i), cumsum(l[2:end]).-l[1]))[2] # which image is closest
                #@show typeof(Gray.([meteosat[mi] out out2]))
            #    FileIO.save(joinpath(@__DIR__, "..", "output$downscale", "all$iter-$i.png"), Gray.([meteosat[mi] out out2]))
                FileIO.save(joinpath(@__DIR__, "..", "output$downscale", "img$iter-$i.png"), Gray.(out))

                if PLOT
                    p1 = surface(out; shading=false, show_axis=false, colormap = :deep)
                    scale!(p1, 1.0, 1.0, 7.5)
                    #rm(joinpath(@__DIR__, "..", "output$downscale", "surf$iter-$i.png"))
                    FileIO.save(joinpath(@__DIR__, "..", "output$downscale", "surf$iter-$i.png"), p1)
                    p2 = surface(out3; shading=false, show_axis=false, colormap = :deep)
                    scale!(p2, 1.0, 1.0, 7.5)
                    #rm(joinpath(@__DIR__, "..", "output$downscale", "surf$iter-$i.png"))
                    FileIO.save(joinpath(@__DIR__, "..", "output$downscale", "surfr$iter-$i.png"), p2)

                end
            end
            run(`ffmpeg -y -r 40 -f image2 -i output$downscale/img$iter-%d.png -vcodec libx264 -crf 25 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p img$downscale-$iter.mp4`)
            PLOT && run(`ffmpeg -y -r 40 -f image2 -i output$downscale/surf$iter-%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p surf$downscale-$iter.mp4`)
            PLOT && run(`ffmpeg -y -r 40 -f image2 -i output$downscale/surfr$iter-%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p surfr$downscale-$iter.mp4`)


        end
    end
end
#run(`convert -delay 10 -resize 400% -loop 0 all{1..60}.png out.gif`)
#  convert -delay 10 -resize 400% -loop 0 output/all3-{1..60}.png out.gif

#ffmpeg -r 60 -f image2 -i output/surf1-%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p surf.mp4

# ffmpeg -i surf2.mp4 -filter:v "setpts=1.5*PTS" surf3.mp4


println("Done.")

# ffmpeg -r 40 -f image2 -i output/img1-%d.png -vcodec libx264 -crf 25 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p img1.mp4
lines(first.(θs));
lines!(last.(θs))
