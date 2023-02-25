module SpinGlassNetworks
    using LabelledGraphs
    using LightGraphs
    using MetaGraphs # TODO: remove that
    using CSV
    using DocStringExtensions
    using LinearAlgebra, MKL
    using Base.Cartesian

    include("ising.jl")
    include("spectrum.jl")
    include("lattice.jl")
    include("factor.jl")
    include("utils.jl")
end # module
