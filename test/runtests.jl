using SpinGlassNetworks
using LabelledGraphs
using Graphs
using MetaGraphs
using Logging
using Test

Base.:(==)(e1::LabelledEdge, e2::LabelledEdge) = src(e1) == src(e2) && dst(e1) == dst(e2)

my_tests = ["ising.jl", "factor.jl"]

for my_test ∈ my_tests
    include(my_test)
end
