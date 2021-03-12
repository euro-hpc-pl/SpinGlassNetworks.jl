using SpinGlassNetworks
using LightGraphs
using MetaGraphs
using Logging

disable_logging(LogLevel(1))

using Test

my_tests = []
push!(my_tests,
      "ising.jl",
      "factor.jl",
)

for my_test in my_tests
    include(my_test)
end