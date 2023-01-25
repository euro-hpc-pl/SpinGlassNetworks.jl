

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

# TODO This is version for only one GPU
function get_projector!(lp::PoolOfProjectors, index::Int, device::Symbol)
    if device ∉ lp.data
        push!(lp.data[device], index => CuArray(lp.data[default_device][index]))
    end
    lp.data[device][index]
end

#=
buf_a = Mem.alloc(Mem.Unified, sizeof(a))
d_a = unsafe_wrap(CuArray{Float32,3}, convert(CuPtr{Float32}, buf_a), dims)
finalizer(d_a) do _
    Mem.free(buf_a)
end
copyto!(d_a, a)
=#

#=
function add_projector!(lp::PoolOfProjectors, p::Proj)
    if p in values(lp.data)
        key = key of p_in_lp.data
    else
        key = generate_new_key
        push!(lp.data, key => p)
    end
    key
end
=#
