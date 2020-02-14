module BridgeSPDE
using Bridge
import Bridge: outer, b, a

using GaussianDistributions, Trajectories, DynamicIterators
using LinearAlgebra, FileIO, Random, Statistics, SparseArrays, Colors

using DynamicIterators: dub
import DynamicIterators: evolve, dyniterate

export gridderiv, gridlaplacian, OrnsteinUhlenbeck, b, riccatti
export downsample, boundary
export Euler
export sparsity

include("sparse.jl")
include("laplace.jl")
include("ornsteinuhlenbeck.jl")

include("sidrs.jl")
include("gaussian.jl")
end # module
