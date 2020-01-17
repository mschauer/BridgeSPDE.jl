
struct OrnsteinUhlenbeck{TB, Tβ, Ta} <: Evolution
    B::TB
    β::Tβ
    a::Ta
end
b(t, x, OU::OrnsteinUhlenbeck) = OU.B*x + OU.β
riccatti(t, P, OU::OrnsteinUhlenbeck) = OU.B*P + P*OU.B + OU.a

struct Euler{T} <: Evolution
    M::T
end

function evolve(M::Euler{<:OrnsteinUhlenbeck}, (s, u)::Pair, t)
    x, P = u
    t => Gaussian(x + b(s, x, M.M)*dt, P + riccatti(t, P, M.M)*dt)
end

function dyniterate(M::Euler{<:OrnsteinUhlenbeck}, (s, u)::Pair, t)
    x, P = u
    dub(t => Gaussian(x + b(s, x, M.M)*dt, P + riccatti(t, P, M.M)*dt))
end

function evolve(M::OrnsteinUhlenbeck, (s, u)::Pair, t)
    x, P = u
    t => Gaussian(x + b(s, x, M.M)*dt, P + riccatti(t, P, M.M)*dt)
end
