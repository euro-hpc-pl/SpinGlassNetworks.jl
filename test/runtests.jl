using SpinGlassNetworks
using LabelledGraphs
using LightGraphs
using MetaGraphs
using Logging
using Test
using TensorCast
using CUDA

Base.:(==)(e1::LabelledEdge, e2::LabelledEdge) = src(e1) == src(e2) && dst(e1) == dst(e2)

function _energy(config::Dict, couplings::Dict, cedges::Dict, n::Int)
    eng = zeros(1,n)
    for (i, j) ∈ keys(cedges)
        for (k, l) ∈ values(cedges[i, j])
            for m ∈ 1:length(config[k])
                s = config[k][m]
                r = config[l][m]
                J = couplings[k, l]
                if k == l
                    eng[m] += dot(s, J)
                else
                    eng[m] += dot(s, J, r)
                end
            end
       end
    end
    eng
end

my_tests = [
    "ising.jl",
    "clustered_hamiltonian.jl",
    "bp_1site.jl",
    "bp_2site.jl",
    "utils.jl",
    "projectors.jl"
    ]

for my_test ∈ my_tests
    include(my_test)
end
