[![Coverage Status](https://coveralls.io/repos/github/iitis/SpinGlassNetworks.jl/badge.svg?branch=master)](https://coveralls.io/github/iitis/SpinGlassNetworks.jl?branch=master)
# SpinGlassNetworks.jl

Package for storing and working with Ising instances and factor graphs.

## Usage

A simple example of a brute force solution for an Ising instance

```julia
using SpinGlassNetworks

function bench(instance::String, size::NTuple{3, Int}, max_states::Int=100)
    ig = ising_graph(instance)
    cl = split_into_clusters(ig, super_square_lattice(size))
    @time sp = brute_force(cl[1, 1], num_states=max_states)
    nothing
end

bench("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt", (2, 2, 24))
```
