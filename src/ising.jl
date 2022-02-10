using LabelledGraphs

export
    ising_graph,
    rank_vec,
    cluster,
    rank,
    nodes,
    basis_size,
    biases,
    couplings,
    IsingGraph,
    prune

const Instance = Union{String, Dict}
const IsingGraph = LabelledGraph{MetaGraph{Int64, Float64}, Int64}

function unique_nodes(ising_tuples)
    sort(collect(Set(Iterators.flatten((i, j) for (i, j, _) ∈ ising_tuples))))
end

"""
Creates Ising graph, H = sgn * sum_{i, j} (J_{ij} * s_i * s_j + J_{ii} * s_i)
"""
function ising_graph(
    instance::Instance; sgn::Real=1.0, rank_override::Dict{Int, Int}=Dict{Int, Int}()
)
    # load the Ising instance
    if instance isa String
        ising = CSV.File(instance, types = [Int, Int, Float64], header=0, comment = "#")
    else
        ising = [(i, j, J) for ((i, j), J) ∈ instance]
    end
    ig = LabelledGraph{MetaGraph}(unique_nodes(ising))

    set_prop!.(Ref(ig), vertices(ig), :h, 0)
    foreach(v -> set_prop!(ig, v, :rank, get(rank_override, v, 2)), vertices(ig))

    for (i, j, v) ∈ ising
        v *= sgn
        if i == j
            set_prop!(ig, i, :h, v)
        else
            add_edge!(ig, i, j) || throw(ArgumentError("Duplicate Egde ($i, $j)"))
            set_prop!(ig, i, j, :J, v)
        end
    end
    set_prop!(
        ig, :rank, Dict{Int, Int}(v => get(rank_override, v, 2) for v in vertices(ig))
    )
    ig
end
rank_vec(ig::IsingGraph) = Int[get_prop((ig), v, :rank) for v ∈ vertices(ig)]
basis_size(ig::IsingGraph) = prod(rank_vec(ig))
biases(ig::IsingGraph) = get_prop.(Ref(ig), vertices(ig), :h)

function couplings(ig::IsingGraph)
    J = zeros(nv(ig), nv(ig))
    for edge ∈ edges(ig)
        i, j = ig.reverse_label_map[src(edge)], ig.reverse_label_map[dst(edge)]
        @inbounds J[i, j] = get_prop(ig, edge, :J)
    end
    J
end
cluster(ig::IsingGraph, verts) = induced_subgraph(ig, collect(verts))

"""
Returns dense adjacency matrix between clusters.
"""
function inter_cluster_edges(ig::IsingGraph, cl1::IsingGraph, cl2::IsingGraph)
    outer_edges = [
        LabelledEdge(i, j) for i ∈ vertices(cl1), j ∈ vertices(cl2) if has_edge(ig, i, j)
    ]
    J = zeros(nv(cl1), nv(cl2))
    for e ∈ outer_edges
        i, j = cl1.reverse_label_map[src(e)], cl2.reverse_label_map[dst(e)]
        @inbounds J[i, j] = get_prop(ig, e, :J)
    end
    outer_edges, J
end

"""
Get rid of non-existing spins.
Used only in MPS_search, would be obsolete if MPS_search uses QMps.
"""
function prune(ig::IsingGraph)
    to_keep = vcat(
        findall(!iszero, degree(ig)),
        findall(
            x -> iszero(degree(ig, x)) && !isapprox(get_prop(ig, x, :h), 0, atol=1e-14),
            vertices(ig)
        )
    )
    gg = ig[ig.labels[to_keep]]
    labels = collect(vertices(gg.inner_graph))
    reverse_label_map = Dict(i => i for i=1:nv(gg.inner_graph))
    LabelledGraph(labels, gg.inner_graph, reverse_label_map)
end
