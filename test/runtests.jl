using SpinGlassNetworks
using SpinGlassTensors
using LabelledGraphs
using Graphs
using MetaGraphs
using Logging
using Test
using CUDA


function _energy(config::Dict, couplings::Dict, cedges::Dict, n::Int)
    eng = zeros(1, n)
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
    "potts_hamiltonian.jl",
    "bp_1site.jl",
    "bp_2site.jl",
    "utils.jl",
    "spectrum.jl",
]

for my_test ∈ my_tests
    include(my_test)
end
