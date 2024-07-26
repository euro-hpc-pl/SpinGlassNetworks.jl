export potts_hamiltonian,
    rank_reveal,
    split_into_clusters,
    decode_potts_hamiltonian_state,
    energy,
    energy_2site,
    cluster_size,
    truncate_potts_hamiltonian,
    exact_cond_prob,
    bond_energy,
    cluster_size

"""
$(TYPEDSIGNATURES)

Group spins into clusters based on an assignment rule, mapping Potts Hamiltonian coordinates to groups of spins in the Ising graph.
Dict(Potts Hamiltonian coordinates -> group of spins in Ising graph)

# Arguments:
- `ig::LabelledGraph{G, L}`: The Ising graph represented as a labeled graph.
- `assignment_rule`: A mapping that assigns Ising graph vertices to clusters based on Potts Hamiltonian coordinates.

# Returns:
- `clusters::Dict{L, Vertex}`: A dictionary mapping cluster identifiers to representative vertices in the Ising graph.

This function groups spins in the Ising graph into clusters based on an assignment rule. 
The assignment rule defines how Potts Hamiltonian coordinates correspond to clusters of spins in the Ising graph. 
Each cluster is represented by a vertex from the Ising graph.

The `split_into_clusters` function is useful for organizing and analyzing spins in complex spin systems, particularly in the context of Potts Hamiltonian.

"""
function split_into_clusters(ig::LabelledGraph{G,L}, assignment_rule) where {G,L}
    cluster_id_to_verts = Dict(i => L[] for i in values(assignment_rule))
    for v in vertices(ig)
        push!(cluster_id_to_verts[assignment_rule[v]], v)
    end
    Dict(i => first(cluster(ig, verts)) for (i, verts) ∈ cluster_id_to_verts)
end

"""
$(TYPEDSIGNATURES)

Create a Potts Hamiltonian.

This function constructs a Potts Hamiltonian from an Ising graph by introducing a natural order in Potts Hamiltonian coordinates.

# Arguments:
- `ig::IsingGraph`: The Ising graph representing the spin system.
- `num_states_cl::Int`: The number of states per cluster taken into account when calculating the spectrum. In every cluster the number of states is constant.
- `spectrum::Function`: A function for calculating the spectrum of the Potts Hamiltonian. It can be `full_spectrum` or `brute_force`.
- `cluster_assignment_rule::Dict{Int, L}`: A dictionary specifying the assignment rule that maps Ising graph vertices to clusters. It can be `super_square_lattice`, `pegasus_lattice` or `zephyr_lattice`.

# Returns:
- `potts_h::LabelledGraph{S, T}`: The Potts Hamiltonian represented as a labelled graph.

The `potts_hamiltonian` function takes an Ising graph (`ig`) as input and constructs a Potts Hamiltonian by 
introducing a natural order in Potts Hamiltonian coordinates. 
It allows you to specify the number of states per cluster, a spectrum calculation function, 
and a cluster assignment rule, which maps Ising graph vertices to clusters.
"""
function potts_hamiltonian(
    ig::IsingGraph,
    num_states_cl::Int;
    spectrum::Function = full_spectrum,
    cluster_assignment_rule::Dict{Int,L}, # e.g. square lattice
) where {L}
    ns = Dict(i => num_states_cl for i ∈ Set(values(cluster_assignment_rule)))
    potts_hamiltonian(
        ig,
        ns,
        spectrum = spectrum,
        cluster_assignment_rule = cluster_assignment_rule,
    )
end

"""
$(TYPEDSIGNATURES)

Create a Potts Hamiltonian.

This function constructs a Potts Hamiltonian from an Ising graph by introducing a natural order in Potts Hamiltonian coordinates.

# Arguments:
- `ig::IsingGraph`: The Ising graph representing the spin system.
- `num_states_cl::Dict{T, Int}`: A dictionary specifying the number of states per cluster for different clusters. Number of states are considered when calculating the spectrum.
- `spectrum::Function`: A function for calculating the spectrum of the Potts Hamiltonian. It can be `full_spectrum` or `brute_force`.
- `cluster_assignment_rule::Dict{Int, T}`: A dictionary specifying the assignment rule that maps Ising graph vertices to clusters. It can be `super_square_lattice`, `pegasus_lattice` or `zephyr_lattice`.

# Returns:
- `potts_h::LabelledGraph{MetaDiGraph}`: The Potts Hamiltonian represented as a labelled graph.

The `potts_hamiltonian` function takes an Ising graph (`ig`) as input and constructs a Potts Hamiltonian 
by introducing a natural order in Potts Hamiltonian coordinates. It allows you to specify the number of 
states per cluster which can vary for different clusters, a spectrum calculation function, 
and a cluster assignment rule, which maps Ising graph vertices to clusters.

"""
function potts_hamiltonian(
    ig::IsingGraph,
    num_states_cl::Dict{T,Int};
    spectrum::Function = full_spectrum,
    cluster_assignment_rule::Dict{Int,T},
) where {T}
    potts_h = LabelledGraph{MetaDiGraph}(sort(unique(values(cluster_assignment_rule))))

    lp = PoolOfProjectors{Int}()

    for (v, cl) ∈ split_into_clusters(ig, cluster_assignment_rule)
        sp = spectrum(cl, num_states = get(num_states_cl, v, basis_size(cl)))
        set_props!(potts_h, v, Dict(:cluster => cl, :spectrum => sp))
    end

    for (i, v) ∈ enumerate(vertices(potts_h)), w ∈ vertices(potts_h)[i+1:end]
        cl1, cl2 = get_prop(potts_h, v, :cluster), get_prop(potts_h, w, :cluster)
        outer_edges, J = inter_cluster_edges(ig, cl1, cl2)

        if !isempty(outer_edges)
            ind1 = any(i -> i != 0, J, dims = 2)
            ind2 = any(i -> i != 0, J, dims = 1)
            ind1 = reshape(ind1, length(ind1))
            ind2 = reshape(ind2, length(ind2))
            JJ = J[ind1, ind2]

            states_v = get_prop(potts_h, v, :spectrum).states
            states_w = get_prop(potts_h, w, :spectrum).states

            pl, unique_states_v = rank_reveal([s[ind1] for s ∈ states_v], :PE)
            pr, unique_states_w = rank_reveal([s[ind2] for s ∈ states_w], :PE)
            en = inter_cluster_energy(unique_states_v, JJ, unique_states_w)
            ipl = add_projector!(lp, pl)
            ipr = add_projector!(lp, pr)

            add_edge!(potts_h, v, w)
            set_props!(
                potts_h,
                v,
                w,
                Dict(:outer_edges => outer_edges, :ipl => ipl, :en => en, :ipr => ipr),
            )
        end
    end
    set_props!(potts_h, Dict(:pool_of_projectors => lp))
    potts_h
end

"""
$(TYPEDSIGNATURES)

Create a Potts Hamiltonian with optional cluster sizes.

This function constructs a Potts Hamiltonian from an Ising graph by introducing a natural order in Potts Hamiltonian coordinates.

# Arguments:
- `ig::IsingGraph`: The Ising graph representing the spin system.
- `spectrum::Function`: A function for calculating the spectrum of the Potts Hamiltonian. It can be `full_spectrum` or `brute_force`. Default is `full_spectrum`.
- `cluster_assignment_rule::Dict{Int, T}`: A dictionary specifying the assignment rule that maps Ising graph vertices to clusters. It can be `super_square_lattice`, `pegasus_lattice` or `zephyr_lattice`.

# Returns:
- `potts_h::LabelledGraph{MetaDiGraph}`: The Potts Hamiltonian represented as a labelled graph.

The `potts_hamiltonian` function takes an Ising graph (`ig`) as input and constructs a Potts Hamiltonian 
by introducing a natural order in Potts Hamiltonian coordinates. 
You can optionally specify a spectrum calculation function and a cluster assignment rule, which maps Ising graph vertices to clusters.
This version of `potts_hamiltonian` function does not truncate states in the cluster while calculating the spectrum.
If you want to specify custom cluster sizes, use the alternative version of this function by 
passing a `Dict{T, Int}` containing the number of states per cluster as `num_states_cl`.
"""
function potts_hamiltonian(
    ig::IsingGraph;
    spectrum::Function = full_spectrum,
    cluster_assignment_rule::Dict{Int,T},
) where {T}
    potts_hamiltonian(
        ig,
        Dict{T,Int}(),
        spectrum = spectrum,
        cluster_assignment_rule = cluster_assignment_rule,
    )
end

# """
# $(TYPEDSIGNATURES)

# Reveal ranks and energies in a specified order.

# This function calculates and reveals the ranks and energies of a set of states in either the
# 'PE' (Projector Energy) or 'EP' (Energy Projector) order.

# # Arguments:
# - `energy`: The energy values of states.
# - `order::Symbol`: The order in which to reveal the ranks and energies. 
# It can be either `:PE` for 'Projector Energy)' order (default) or `:EP` for 'Energy Projector' order.

# # Returns:
# - If `order` is `:PE`, the function returns a tuple `(P, E)` where:
#   - `P`: A permutation matrix representing projectors.
#   - `E`: An array of energy values.
# - If `order` is `:EP`, the function returns a tuple `(E, P)` where:
#   - `E`: An array of energy values.
#   - `P`: A permutation matrix representing projectors.
# """
# function rank_reveal(energy, order=:PE) #TODO: add type
#     @assert order ∈ (:PE, :EP)
#     dim = order == :PE ? 1 : 2
#     E, idx = unique_dims(energy, dim)
#     P = identity.(idx)
#     order == :PE ? (P, E) : (E, P)
# end

"""
$(TYPEDSIGNATURES)

TODO: check the order consistency over external packages.

Decode a Potts Hamiltonian state into Ising graph spin values.

This function decodes a state from a Potts Hamiltonian into Ising graph spin values and 
returns a dictionary mapping each Ising graph vertex to its corresponding spin value.

# Arguments:
- `potts_h::LabelledGraph{S, T}`: The Potts Hamiltonian represented as a labeled graph.
- `state::Vector{Int}`: The state to be decoded, represented as an array of state indices for each vertex in the Potts Hamiltonian.

# Returns:
- `spin_values::Dict{Int, Int}`: A dictionary mapping each Ising graph vertex to its corresponding spin value.

This function assumes that the state has the same order as the vertices in the Potts Hamiltonian. 
It decodes the state consistently based on the cluster assignments and spectra of the Potts Hamiltonian.
"""
function decode_potts_hamiltonian_state(
    potts_h::LabelledGraph{S,T},
    state::Vector{Int},
) where {S,T}
    ret = Dict{Int,Int}()
    for (i, vert) ∈ zip(state, vertices(potts_h))
        spins = get_prop(potts_h, vert, :cluster).labels
        states = get_prop(potts_h, vert, :spectrum).states
        if length(states) > 0
            curr_state = states[i]
            merge!(ret, Dict(k => v for (k, v) ∈ zip(spins, curr_state)))
        end
    end
    ret
end

"""
$(TYPEDSIGNATURES)

Calculate the energy of a Potts Hamiltonian state.

This function calculates the energy of a given state in a Potts Hamiltonian. 
The state is represented as a dictionary mapping each Ising graph vertex to its corresponding spin value.

# Arguments:
- `potts_h::LabelledGraph{S, T}`: The Potts Hamiltonian represented as a labeled graph.
- `σ::Dict{T, Int}`: A dictionary mapping Ising graph vertices to their spin values.

# Returns:
- `en_potts_h::Float64`: The energy of the state in the Potts Hamiltonian.

This function computes the energy by summing the energies associated with individual 
clusters and the interaction energies between clusters. 
It takes into account the cluster spectra and projectors stored in the Potts Hamiltonian.
"""
function energy(potts_h::LabelledGraph{S,T}, σ::Dict{T,Int}) where {S,T}
    en_potts_h = 0.0
    for v ∈ vertices(potts_h)
        en_potts_h += get_prop(potts_h, v, :spectrum).energies[σ[v]]
    end
    for edge ∈ edges(potts_h)
        idx_pl = get_prop(potts_h, edge, :ipl)
        pl = get_projector!(get_prop(potts_h, :pool_of_projectors), idx_pl, :CPU)
        idx_pr = get_prop(potts_h, edge, :ipr)
        pr = get_projector!(get_prop(potts_h, :pool_of_projectors), idx_pr, :CPU)
        en = get_prop(potts_h, edge, :en)
        en_potts_h += en[pl[σ[src(edge)]], pr[σ[dst(edge)]]]
    end
    en_potts_h
end

"""
$(TYPEDSIGNATURES)

Calculate the interaction energy between two nodes in a Potts Hamiltonian.

This function computes the interaction energy between two specified nodes in a Potts Hamiltonian, represented as a labeled graph.

# Arguments:
- `potts_h::LabelledGraph{S, T}`: The Potts Hamiltonian represented as a labeled graph.
- `i::Int`: The index of the first site.
- `j::Int`: The index of the second site.

# Returns:
- `int_eng::AbstractMatrix{T}`: The interaction energy matrix between the specified sites.

The function checks if there is an interaction edge between the two sites (i, j) in both directions (i -> j and j -> i). 
If such edges exist, it retrieves the interaction energy matrix, projectors, and calculates the interaction energy. 
If no interaction edge is found, it returns a zero matrix.
"""
function energy_2site(potts_h::LabelledGraph{S,T}, i::Int, j::Int) where {S,T}
    # matrix of interaction energies between two nodes
    if has_edge(potts_h, (i, j, 1), (i, j, 2))
        en12 = copy(get_prop(potts_h, (i, j, 1), (i, j, 2), :en))
        idx_pl = get_prop(potts_h, (i, j, 1), (i, j, 2), :ipl)
        pl = copy(get_projector!(get_prop(potts_h, :pool_of_projectors), idx_pl, :CPU))
        idx_pr = get_prop(potts_h, (i, j, 1), (i, j, 2), :ipr)
        pr = copy(get_projector!(get_prop(potts_h, :pool_of_projectors), idx_pr, :CPU))
        int_eng = en12[pl, pr]
    elseif has_edge(potts_h, (i, j, 2), (i, j, 1))
        en21 = copy(get_prop(potts_h, (i, j, 2), (i, j, 1), :en))
        idx_pl = get_prop(potts_h, (i, j, 2), (i, j, 1), :ipl)
        pl = copy(get_projector!(get_prop(potts_h, :pool_of_projectors), idx_pl, :CPU))
        idx_pr = get_prop(potts_h, (i, j, 2), (i, j, 1), :ipr)
        pr = copy(get_projector!(get_prop(potts_h, :pool_of_projectors), idx_pr, :CPU))
        int_eng = en21[pl, pr]'
    else
        int_eng = zeros(1, 1)
    end
    int_eng
end

"""
$(TYPEDSIGNATURES)

Calculate the bond energy between two clusters in a Potts Hamiltonian.

This function computes the bond energy between two specified clusters (cluster nodes) in a Potts Hamiltonian, represented as a labeled graph.

# Arguments:
- `potts_h::LabelledGraph{S, T}`: The Potts Hamiltonian represented as a labeled graph.
- `potts_h_u::NTuple{N, Int64}`: The coordinates of the first cluster.
- `potts_h_v::NTuple{N, Int64}`: The coordinates of the second cluster.
- `σ::Int`: Index for which the bond energy is calculated.

# Returns:
- `energies::AbstractVector{T}`: The bond energy vector between the two clusters for the specified index.

The function checks if there is an edge between the two clusters (u -> v and v -> u). 
If such edges exist, it retrieves the bond energy matrix and projectors and calculates the bond energy. 
If no bond edge is found, it returns a zero vector.
"""
function bond_energy(
    potts_h::LabelledGraph{S,T},
    potts_h_u::NTuple{N,Int64},
    potts_h_v::NTuple{N,Int64},
    σ::Int,
) where {S,T,N}
    if has_edge(potts_h, potts_h_u, potts_h_v)
        ipu, en, ipv = get_prop.(Ref(potts_h), Ref(potts_h_u), Ref(potts_h_v), (:ipl, :en, :ipr))
        pu = get_projector!(get_prop(potts_h, :pool_of_projectors), ipu, :CPU)
        pv = get_projector!(get_prop(potts_h, :pool_of_projectors), ipv, :CPU)
        @inbounds energies = en[pu, pv[σ]]
    elseif has_edge(potts_h, potts_h_v, potts_h_u)
        ipv, en, ipu = get_prop.(Ref(potts_h), Ref(potts_h_v), Ref(potts_h_u), (:ipl, :en, :ipr))
        pu = get_projector!(get_prop(potts_h, :pool_of_projectors), ipu, :CPU)
        pv = get_projector!(get_prop(potts_h, :pool_of_projectors), ipv, :CPU)
        @inbounds energies = en[pv[σ], pu]
    else
        energies = zeros(cluster_size(potts_h, potts_h_u))
    end
end

"""
$(TYPEDSIGNATURES)

Get the size of a cluster in a Potts Hamiltonian.

This function returns the size (number of states) of a cluster in a Potts Hamiltonian, represented as a labeled graph.

# Arguments:
- `potts_hamiltonian::LabelledGraph{S, T}`: The Potts Hamiltonian represented as a labeled graph.
- `vertex::T`: The vertex (cluster) for which the size is to be determined.

# Returns:
- `size::Int`: The number of states in the specified cluster.

The function retrieves the spectrum associated with the specified cluster and returns the length of the energy vector in that spectrum.
"""
function cluster_size(potts_hamiltonian::LabelledGraph{S,T}, vertex::T) where {S,T}
    length(get_prop(potts_hamiltonian, vertex, :spectrum).energies)
end

"""
$(TYPEDSIGNATURES)

Calculate the exact conditional probability of a target state in a Potts Hamiltonian.

This function computes the exact conditional probability of a specified target state in a Potts Hamiltonian, represented as a labelled graph.

# Arguments:
- `potts_hamiltonian::LabelledGraph{S, T}`: The Potts Hamiltonian represented as a labeled graph.
- `beta`: The inverse temperature parameter.
- `target_state::Dict`: A dictionary specifying the target state as a mapping of cluster vertices to Ising spin values.

# Returns:
- `prob::Float64`: The exact conditional probability of the target state.

The function generates all possible states for the clusters in the Potts Hamiltonian, 
calculates their energies, and computes the probability distribution based on the given inverse temperature parameter. 
It then calculates the conditional probability of the specified target state by summing the probabilities of states that match the target state.
"""
function exact_cond_prob(
    potts_hamiltonian::LabelledGraph{S,T},
    beta,
    target_state::Dict,
) where {S,T}
    # TODO: Not going to work without PoolOfProjectors
    ver = vertices(potts_hamiltonian)
    rank = cluster_size.(Ref(potts_hamiltonian), ver)
    states = [Dict(ver .=> σ) for σ ∈ Iterators.product([1:r for r ∈ rank]...)]
    energies = SpinGlassNetworks.energy.(Ref(potts_hamiltonian), states)
    prob = exp.(-beta .* energies)
    prob ./= sum(prob)
    sum(prob[findall([all(s[k] == v for (k, v) ∈ target_state) for s ∈ states])])
end

"""
$(TYPEDSIGNATURES)

Truncate a Potts Hamiltonian based on specified states.

This function truncates a given Potts Hamiltonian by selecting a subset of states for each cluster based on the provided `states` dictionary. 
The resulting truncated Hamiltonian contains only the selected states for each cluster.

# Arguments:
- `potts_h::LabelledGraph{S, T}`: The Potts Hamiltonian represented as a labeled graph.
- `states::Dict`: A dictionary specifying the states to be retained for each cluster.

# Returns:
- `new_potts_h::LabelledGraph{MetaDiGraph}`: The truncated Potts Hamiltonian with reduced states.

The function creates a new Potts Hamiltonian `new_potts_h` with the same structure as the input `potts_h`. 
It then updates the spectrum of each cluster in `new_potts_h` by selecting the specified states from the original spectrum. 
Additionally, it updates the interactions and projectors between clusters based on the retained states. 
The resulting `new_potts_h` represents a truncated version of the original Hamiltonian.
"""
function truncate_potts_hamiltonian(potts_h::LabelledGraph{S,T}, states::Dict) where {S,T}

    new_potts_h = LabelledGraph{MetaDiGraph}(vertices(potts_h))
    new_lp = PoolOfProjectors{Int}()

    for v ∈ vertices(new_potts_h)
        cl = get_prop(potts_h, v, :cluster)
        sp = get_prop(potts_h, v, :spectrum)
        if sp.states == Vector{Int64}[]
            sp = Spectrum(sp.energies[states[v]], sp.states, [1])
        else
            sp = Spectrum(sp.energies[states[v]], sp.states[states[v]])
        end
        set_props!(new_potts_h, v, Dict(:cluster => cl, :spectrum => sp))
    end

    for e ∈ edges(potts_h)
        v, w = src(e), dst(e)
        add_edge!(new_potts_h, v, w)
        outer_edges = get_prop(potts_h, v, w, :outer_edges)
        ipl = get_prop(potts_h, v, w, :ipl)
        pl = get_projector!(get_prop(potts_h, :pool_of_projectors), ipl, :CPU)
        ipr = get_prop(potts_h, v, w, :ipr)
        pr = get_projector!(get_prop(potts_h, :pool_of_projectors), ipr, :CPU)
        en = get_prop(potts_h, v, w, :en)
        pl = pl[states[v]]
        pr = pr[states[w]]
        pl_transition, pl_unique = rank_reveal(pl, :PE)
        pr_transition, pr_unique = rank_reveal(pr, :PE)
        en = en[pl_unique, pr_unique]
        ipl = add_projector!(new_lp, pl_transition)
        ipr = add_projector!(new_lp, pr_transition)
        set_props!(
            new_potts_h,
            v,
            w,
            Dict(:outer_edges => outer_edges, :ipl => ipl, :en => en, :ipr => ipr),
        )
    end
    set_props!(new_potts_h, Dict(:pool_of_projectors => new_lp))

    new_potts_h
end

function potts_hamiltonian(
    fname::String,
    Nx::Union{Integer,Nothing} = nothing,
    Ny::Union{Integer,Nothing} = nothing,
)

    loaded_rmf = load_openGM(fname, Nx, Ny)
    functions = loaded_rmf["fun"]
    factors = loaded_rmf["fac"]
    N = loaded_rmf["N"]

    X, Y = loaded_rmf["Nx"], loaded_rmf["Ny"]

    clusters = super_square_lattice((X, Y, 1))
    potts_h = LabelledGraph{MetaDiGraph}(sort(collect(values(clusters))))
    lp = PoolOfProjectors{Int}()
    for v ∈ potts_h.labels
        set_props!(potts_h, v, Dict(:cluster => v))
    end
    for (index, value) in factors
        if length(index) == 2
            y, x = index
            Eng = functions[value]'
            sp = Spectrum(collect(Eng), Vector{Vector{Int}}[], zeros(Int, N[y+1, x+1]))
            set_props!(potts_h, (x + 1, y + 1), Dict(:spectrum => sp))
        elseif length(index) == 4
            y1, x1, y2, x2 = index
            add_edge!(potts_h, (x1 + 1, y1 + 1), (x2 + 1, y2 + 1))
            Eng = functions[value]
            ipl = add_projector!(lp, collect(1:N[y1+1, x1+1]))
            ipr = add_projector!(lp, collect(1:N[y2+1, x2+1]))
            set_props!(
                potts_h,
                (x1 + 1, y1 + 1),
                (x2 + 1, y2 + 1),
                Dict(
                    :outer_edges => ((x1 + 1, y1 + 1), (x2 + 1, y2 + 1)),
                    :en => Eng,
                    :ipl => ipl,
                    :ipr => ipr,
                ),
            )
        else
            throw(
                ErrorException(
                    "Something is wrong with factor index, it has length $(length(index))",
                ),
            )
        end
    end

    set_props!(potts_h, Dict(:pool_of_projectors => lp, :Nx => X, :Ny => Y))
    potts_h
end
