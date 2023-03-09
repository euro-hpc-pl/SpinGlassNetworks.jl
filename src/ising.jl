using LabelledGraphs

export ising_graph,
    rank_vec, cluster, rank, nodes, basis_size, biases, couplings, IsingGraph, prune

const Instance = Union{String,Dict}


unique_nodes(ising_tuples) =
    sort(collect(Set(Iterators.flatten((i, j) for (i, j, _) ∈ ising_tuples))))



const IsingGraph = LabelledGraph{MetaGraph{Int64,Float64},Int64}

"""
    ising_graph(instance, sgn::Number = 1.0, rank_override::Dict{Int,Int} = Dict{Int,Int}())

Create an Ising model graph from an input instance. The in

"""
function ising_graph(
    instance::Instance;
    sgn::Number = 1.0,
    rank_override::Dict{Int,Int} = Dict{Int,Int}(),
)
    # load the Ising instance
    if instance isa String
        ising = CSV.File(instance, types = [Int, Int, Float64], header = 0, comment = "#")
    else
        ising = [(i, j, J) for ((i, j), J) ∈ instance]
    end

    nodes = unique_nodes(ising)
    L = length(nodes)
    nodes_to_vertices = Dict(w => i for (i, w) ∈ enumerate(nodes))

    ig = LabelledGraph{MetaGraph}(nodes)

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


    set_prop!(ig, :rank, Dict{Int,Int}(v => get(rank_override, v, 2) for v in vertices(ig)))

    ig
end

"""
    rank_vec(ising_graph)

Return vector of 'ranks' of nodes. In this context rank tels us how many spins values node can have. 
"""
rank_vec(ig::IsingGraph) = Int[get_prop((ig), v, :rank) for v ∈ vertices(ig)]
basis_size(ig::IsingGraph) = prod(prod(rank_vec(ig)))

"""
    biases(ising_graph)

Return vector of biases of ising model defined by given 'ising_graph'.
"""
biases(ig::IsingGraph) = get_prop.(Ref(ig), vertices(ig), :h)

"""
    couplings(ising_graph)

Return couplings of ising model defined by given 'ising_graph'. 
The couplings are presented as an uper-triangular matrix 
"""
function couplings(ig::IsingGraph)
    J = zeros(nv(ig), nv(ig))
    for edge in edges(ig)
        i, j = ig.reverse_label_map[src(edge)], ig.reverse_label_map[dst(edge)]
        J[i, j] = get_prop(ig, edge, :J)
    end
    J
end

cluster(ig::IsingGraph, verts) = induced_subgraph(ig, collect(verts))

function inter_cluster_edges(ig::IsingGraph, cl1::IsingGraph, cl2::IsingGraph)
    outer_edges =
        [LabelledEdge(i, j) for i ∈ vertices(cl1), j ∈ vertices(cl2) if has_edge(ig, i, j)]

    J = zeros(nv(cl1), nv(cl2))
    for e ∈ outer_edges
        i, j = cl1.reverse_label_map[src(e)], cl2.reverse_label_map[dst(e)]
        @inbounds J[i, j] = get_prop(ig, e, :J)
    end
    outer_edges, J
end


"""
    prune(ising_graph)

Return copy of 'ising_graph' with removed isolated vertices.
"""
function prune(ig::IsingGraph)
    to_keep = vcat(
        findall(!iszero, degree(ig)),
        findall(
            x ->
                iszero(degree(ig, x)) &&
                    !isapprox(get_prop(ig, x, :h), 0, atol = 1e-14),
            vertices(ig),
        ),
    )

    gg = ig[ig.labels[to_keep]]
    labels = collect(vertices(gg.inner_graph))
    reverse_label_map = Dict(i => i for i = 1:nv(gg.inner_graph))
    LabelledGraph(labels, gg.inner_graph, reverse_label_map)
end
