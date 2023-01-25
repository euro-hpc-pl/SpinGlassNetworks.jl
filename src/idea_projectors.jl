

export
    PoolOfProjectors

const Proj{T} = Union{Vector{T}, CuArray{T, 1}}

struct PoolOfProjectors{T}
    data::Dict{Symbol, Dict{Int, Proj{T}}}
    default_device::Symbol

    PoolOfProjectors(data:::Dict{Int, Dict{Int, Vector{T}}}) where T = new{T}(data, :CPU)
end

Base.eltype(lp::PoolOfProjectors{T}) where T = T
Base.length(lp::PoolOfProjectors, index::Int) = length(lp.data[lp.default_device][index])
Base.empty(lp::PoolOfProjectors, device::Symbol) = empty!(lp.data[device])

function get_projector!(lp::PoolOfProjectors, index::Int, device::Symbol)

    # sprawdz czy index in lp.data[device]; jesli nie to go dodaj biorac dane z lp.data[default_device]
    lp.data[device][index]  #  dodac element do dataGPU jak jest na cpu, ale nie na gpu ???
end
