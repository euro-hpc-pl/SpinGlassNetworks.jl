# Introduction
A Potts Hamiltonian is a graphical representation that allows for a convenient and intuitive way to describe the structure of a network.

The concept of a Potts Hamiltonian within `SpinGlassNetworks.jl` introduces a mechanism for organizing spins into desired geometries, facilitating a structured approach to modeling complex spin systems. Analogous to a standard factor graph, the Potts Hamiltonian involves nodes that represent tensors within the underlying network. The edges connecting these nodes in the Potts Hamiltonian correspond to the indices shared between the respective tensors in the tensor network.

```@docs
potts_hamiltonian
```

## Simple example

```@example
using SpinGlassNetworks

# Prepare simple instance
instance = "$(@__DIR__)/../../src/instances/square_diagonal/5x5/diagonal.txt"
ig = ising_graph(instance)

# Create Potts Hamiltonian
potts_h = potts_hamiltonian(
    ig,
    cluster_assignment_rule = super_square_lattice((5,5,4))
)
```