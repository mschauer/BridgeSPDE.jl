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

include("laplace.jl")
include("ornsteinuhlenbeck.jl")
end # module
