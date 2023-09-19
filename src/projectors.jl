export
    PoolOfProjectors2,
    get_projector2!,
    add_projector2!

const Proj{T} = Union{Vector{T}, Array{T, 1}}

struct PoolOfProjectors2{T <: Integer}
    data::Dict{Symbol, Dict{Int, Proj{T}}}
    default_device::Symbol
    sizes::Dict{Int, Int}

    PoolOfProjectors2(data::Dict{Int, Dict{Int, Vector{T}}}) where T = new{T}(Dict(:CPU => data),
                                                                            :CPU,
                                                                            Dict{Int, Int}(k => maximum(v) for (k, v) ∈ data))
    PoolOfProjectors2{T}() where T = new{T}(Dict(:CPU => Dict{Int, Proj{T}}()),
                                           :CPU,
                                           Dict{Int, Int}())
end


Base.eltype(lp::PoolOfProjectors2{T}) where T = T
Base.length(lp::PoolOfProjectors2) = length(lp.data[lp.default_device])
Base.length(lp::PoolOfProjectors2, device::Symbol) = length(lp.data[device])

function Base.empty!(lp::PoolOfProjectors2, device::Symbol) 
    if device ∈ keys(lp.data)
        empty!(lp.data[device])
    end
end

Base.length(lp::PoolOfProjectors2, index::Int) = length(lp.data[lp.default_device][index])
Base.size(lp::PoolOfProjectors2, index::Int) = lp.sizes[index]

get_projector2!(lp::PoolOfProjectors2, index::Int) = get_projector2!(lp, index, lp.default_device)

# TODO This is version for only one GPU
function get_projector2!(lp::PoolOfProjectors2{T}, index::Int, device::Symbol) where T <: Integer
    if device ∉ keys(lp.data)
        push!(lp.data, device => Dict{Int, Proj{T}}())
    end

    if index ∉ keys(lp.data[device])
        if device == :GPU
            p = CuArray{T}(lp.data[lp.default_device][index])
        elseif device == :CPU
            p = Array{T}(lp.data[lp.default_device][index])
        else
            throw(ArgumentError("device should be :CPU or :GPU"))
        end
        push!(lp.data[device], index => p)
    end
    lp.data[device][index]
end

function add_projector2!(lp::PoolOfProjectors2{T}, p::Proj) where T <: Integer
    if lp.default_device == :CPU
        p = Array{T}(p)
    elseif lp.default_device == :GPU
        p = CuArray{T}(p)
    else
        throw(ArgumentError("default_device should be :CPU or :GPU"))
    end
    if p in values(lp.data[lp.default_device])
        key = -1
        for guess in keys(lp.data[lp.default_device])
            if lp.data[lp.default_device][guess] == p
                key = guess
                break
            end
        end
    else
        key = length(lp.data[lp.default_device]) + 1
        push!(lp.data[lp.default_device], key => p)
        push!(lp.sizes, key => maximum(p))
    end
    key
end
