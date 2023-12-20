module SpinGlassNetworks
    using LabelledGraphs
    using LightGraphs
    using MetaGraphs # TODO: to be replaced by MetaGraphsNext
    using CSV
    using DocStringExtensions
    using LinearAlgebra, MKL
    using Base.Cartesian
    using SparseArrays
    using HDF5
    using CUDA, CUDA.CUSPARSE
    import Base.Prehashed


    include("ising.jl")
    include("spectrum.jl")
    include("lattice.jl")
    #include("projectors.jl")
    include("clustered_hamiltonian.jl")
    include("bp.jl")
    include("truncate.jl")
    include("utils.jl")
end # module
