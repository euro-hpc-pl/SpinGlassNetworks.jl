export factor_graph, rank_reveal, projectors, split_into_clusters, decode_factor_graph_state


function split_into_clusters(ig::LabelledGraph{S,T}, assignment_rule) where {S,T}
    cluster_id_to_verts = Dict(i => T[] for i in values(assignment_rule))

    for v in vertices(ig)
        push!(cluster_id_to_verts[assignment_rule[v]], v)
    end

    Dict(i => first(cluster(ig, verts)) for (i, verts) ∈ cluster_id_to_verts)
end


function factor_graph(
    ig::IsingGraph,
    num_states_cl::Int;
    spectrum::Function = full_spectrum,
    cluster_assignment_rule::Dict{Int,T}, # e.g. square lattice
) where {T}
    ns = Dict(i => num_states_cl for i ∈ Set(values(cluster_assignment_rule)))
    factor_graph(
        ig,
        ns,
        spectrum = spectrum,
        cluster_assignment_rule = cluster_assignment_rule,
    )
end

function factor_graph(
    ig::IsingGraph,
    num_states_cl::Dict{T,Int};
    spectrum::Function = full_spectrum,
    cluster_assignment_rule::Dict{Int,T}, # e.g. square lattice
) where {T}
    L = maximum(values(cluster_assignment_rule))
    fg = LabelledGraph{MetaDiGraph}(sort(unique(values(cluster_assignment_rule))))

    for (v, cl) ∈ split_into_clusters(ig, cluster_assignment_rule)
        sp = spectrum(cl, num_states = get(num_states_cl, v, basis_size(cl)))
        set_props!(fg, v, Dict(:cluster => cl, :spectrum => sp))
    end

    for (i, v) ∈ enumerate(vertices(fg)), w ∈ vertices(fg)[i+1:end]
        cl1, cl2 = get_prop(fg, v, :cluster), get_prop(fg, w, :cluster)

        outer_edges, J = inter_cluster_edges(ig, cl1, cl2)

        if !isempty(outer_edges)
            en = inter_cluster_energy(
                get_prop(fg, v, :spectrum).states,
                J,
                get_prop(fg, w, :spectrum).states,
            )
            pl, en = rank_reveal(en, :PE)
            en, pr = rank_reveal(en, :EP)
            add_edge!(fg, v, w)
            set_props!(
                fg,
                v,
                w,
                Dict(:outer_edges => outer_edges, :pl => pl, :en => en, :pr => pr),
            )
        end
    end
    fg
end

function factor_graph(
    ig::IsingGraph;
    spectrum::Function = full_spectrum,
    cluster_assignment_rule::Dict{Int,T},
) where {T}
    factor_graph(
        ig,
        Dict{T,Int}(),
        spectrum = spectrum,
        cluster_assignment_rule = cluster_assignment_rule,
    )
end

function rank_reveal(energy, order = :PE)
    @assert order ∈ (:PE, :EP)
    dim = order == :PE ? 1 : 2

    E, idx = unique_dims(energy, dim)

    if order == :PE
        P = zeros(size(energy, 1), size(E, 1))
    else
        P = zeros(size(E, 2), size(energy, 2))
    end

    for (i, elements) ∈ enumerate(eachslice(P, dims = dim))
        elements[idx[i]] = 1
    end

    order == :PE ? (P, E) : (E, P)
end


function decode_factor_graph_state(fg, state::Vector{Int})
    ret = Dict{Int,Int}()
    for (i, vert) ∈ zip(state, vertices(fg))
        spins = get_prop(fg, vert, :cluster).labels
        states = get_prop(fg, vert, :spectrum).states
        if length(states) > 0
            curr_state = states[i]
            merge!(ret, Dict(k => v for (k, v) ∈ zip(spins, curr_state)))
        end
    end
    ret
end
