using SpinGlassNetworks
using LabelledGraphs
using LightGraphs
using MetaGraphs
using Logging
using Test

Base.:(==)(e1::LabelledEdge, e2::LabelledEdge) = src(e1) == src(e2) && dst(e1) == dst(e2)

idx(σ::Int) = (σ == -1) ? 1 : σ + 1
energy(σ::Vector, J::Matrix, η::Vector=σ) = dot(σ, J, η)
energy(σ::Vector, h::Vector) = dot(h, σ)
energy(σ::Vector, ig::IsingGraph) = energy(σ, couplings(ig)) + energy(σ, biases(ig))

my_tests = ["ising.jl", "factor.jl"]

for my_test ∈ my_tests
    include(my_test)
end
