export
    clustered_hamiltonian,
    rank_reveal,
    split_into_clusters,
    decode_clustered_hamiltonian_state,
    energy,
    energy_2site,
    cluster_size,
    truncate_clustered_hamiltonian,
    exact_cond_prob,
    bond_energy

"""
Groups spins into clusters: Dict(factor graph coordinates -> group of spins in Ising graph)
"""

function split_into_clusters(ig::LabelledGraph{G, L}, assignment_rule) where {G, L}
    cluster_id_to_verts = Dict(i => L[] for i in values(assignment_rule))
    for v in vertices(ig) push!(cluster_id_to_verts[assignment_rule[v]], v) end
    Dict(i => first(cluster(ig, verts)) for (i, verts) ∈ cluster_id_to_verts)
end

"""
Create factor graph.
Factor graph order introduced as a natural order in factor graph coordinates.
"""
function clustered_hamiltonian(
    ig::IsingGraph,
    num_states_cl::Int;
    spectrum::Function=full_spectrum,
    cluster_assignment_rule::Dict{Int, L} # e.g. square lattice
) where L
    ns = Dict(i => num_states_cl for i ∈ Set(values(cluster_assignment_rule)))
    clustered_hamiltonian(ig, ns, spectrum=spectrum, cluster_assignment_rule=cluster_assignment_rule)
end


function clustered_hamiltonian(
    ig::IsingGraph,
    num_states_cl::Dict{T, Int};
    spectrum::Function=full_spectrum,
    cluster_assignment_rule::Dict{Int, T}
) where T
    cl_h = LabelledGraph{MetaDiGraph}(
        sort(unique(values(cluster_assignment_rule)))
    )

    lp = PoolOfProjectors{Int}()

    for (v, cl) ∈ split_into_clusters(ig, cluster_assignment_rule)
        sp = spectrum(cl, num_states=get(num_states_cl, v, basis_size(cl)))
        set_props!(cl_h, v, Dict(:cluster => cl, :spectrum => sp))
    end

    for (i, v) ∈ enumerate(vertices(cl_h)), w ∈ vertices(cl_h)[i+1:end]
        cl1, cl2 = get_prop(cl_h, v, :cluster), get_prop(cl_h, w, :cluster)
        outer_edges, J = inter_cluster_edges(ig, cl1, cl2)

        if !isempty(outer_edges)
            ind1 = any(i -> i != 0, J, dims=2)
            ind2 = any(i -> i != 0, J, dims=1)
            ind1 = reshape(ind1, length(ind1))
            ind2 = reshape(ind2, length(ind2))
            JJ = J[ind1, ind2]

            states_v = get_prop(cl_h, v, :spectrum).states
            states_w = get_prop(cl_h, w, :spectrum).states

            pl, unique_states_v = rank_reveal([s[ind1] for s ∈ states_v], :PE)
            pr, unique_states_w = rank_reveal([s[ind2] for s ∈ states_w], :PE)
            en = inter_cluster_energy(unique_states_v, JJ, unique_states_w)
            ipl = add_projector!(lp, pl)
            ipr = add_projector!(lp, pr)

            add_edge!(cl_h, v, w)
            set_props!(
                cl_h, v, w, Dict(:outer_edges => outer_edges, :ipl => ipl, :en => en, :ipr => ipr)
            )
        end
    end
    set_props!(cl_h, Dict(:pool_of_projectors => lp))
    cl_h
end

function clustered_hamiltonian(ig::IsingGraph; spectrum::Function=full_spectrum, cluster_assignment_rule::Dict{Int, T}) where T
    clustered_hamiltonian(ig, Dict{T, Int}(), spectrum=spectrum, cluster_assignment_rule=cluster_assignment_rule)
end

function rank_reveal(energy, order=:PE) where T <: Real #TODO: add type
    @assert order ∈ (:PE, :EP)
    dim = order == :PE ? 1 : 2
    E, idx = unique_dims(energy, dim)
    P = identity.(idx)
    order == :PE ? (P, E) : (E, P)
end

"""
Returns Dict(vertex of ising graph -> spin value)
Assumes that state has the same order as vertices in factor graph!
TODO: check the order consistency over external packages.
"""
function decode_clustered_hamiltonian_state(cl_h::LabelledGraph{S, T}, state::Vector{Int}) where {S, T}
    ret = Dict{Int, Int}()
    for (i, vert) ∈ zip(state, vertices(cl_h))
        spins = get_prop(cl_h, vert, :cluster).labels
        states = get_prop(cl_h, vert, :spectrum).states
        if length(states) > 0
            curr_state = states[i]
            merge!(ret, Dict(k => v for (k, v) ∈ zip(spins, curr_state)))
        end
    end
    ret
end


function energy(cl_h::LabelledGraph{S, T}, σ::Dict{T, Int}) where {S, T}
    en_cl_h = 0.0
    for v ∈ vertices(cl_h) en_cl_h += get_prop(cl_h, v, :spectrum).energies[σ[v]] end
    for edge ∈ edges(cl_h)
        idx_pl = get_prop(cl_h, edge, :ipl)
        pl = get_projector!(get_prop(cl_h, :pool_of_projectors), idx_pl, :CPU)
        idx_pr = get_prop(cl_h, edge, :ipr)
        pr = get_projector!(get_prop(cl_h, :pool_of_projectors), idx_pr, :CPU)
        en = get_prop(cl_h, edge, :en)
        en_cl_h += en[pl[σ[src(edge)]], pr[σ[dst(edge)]]]
    end
    en_cl_h
end


function energy_2site(cl_h::LabelledGraph{S, T}, i::Int, j::Int) where {S, T}
    # matrix of interaction energies between two nodes
    if has_edge(cl_h, (i, j, 1), (i, j, 2))
        en12 = copy(get_prop(cl_h, (i, j, 1), (i, j, 2), :en))
        idx_pl = get_prop(cl_h, (i, j, 1), (i, j, 2), :ipl)
        pl = copy(get_projector!(get_prop(cl_h, :pool_of_projectors), idx_pl, :CPU))
        idx_pr = get_prop(cl_h, (i, j, 1), (i, j, 2), :ipr)
        pr = copy(get_projector!(get_prop(cl_h, :pool_of_projectors), idx_pr, :CPU))
        int_eng = en12[pl, pr]
    elseif has_edge(cl_h, (i, j, 2), (i, j, 1))
        en21 = copy(get_prop(cl_h, (i, j, 2), (i, j, 1), :en))
        idx_pl = get_prop(cl_h, (i, j, 2), (i, j, 1), :ipl)
        pl = copy(get_projector!(get_prop(cl_h, :pool_of_projectors), idx_pl, :CPU))
        idx_pr = get_prop(cl_h, (i, j, 2), (i, j, 1), :ipr)
        pr = copy(get_projector!(get_prop(cl_h, :pool_of_projectors), idx_pr, :CPU))
        int_eng = en21[pl, pr]'
    else
        int_eng = zeros(1, 1)
    end
    int_eng
end

function bond_energy(cl_h::LabelledGraph{S, T}, cl_h_u::NTuple{N, Int64}, cl_h_v::NTuple{N, Int64}, σ::Int) where {S, T, N}
    if has_edge(cl_h, cl_h_u, cl_h_v)
        ipu, en, ipv = get_prop.(
                        Ref(cl_h), Ref(cl_h_u), Ref(cl_h_v), (:ipl, :en, :ipr)
                    )
        pu = get_projector!(get_prop(cl_h, :pool_of_projectors), ipu, :CPU)
        pv = get_projector!(get_prop(cl_h, :pool_of_projectors), ipv, :CPU)
        @inbounds energies = en[pu, pv[σ]]
    elseif has_edge(cl_h, cl_h_v, cl_h_u)
        ipv, en, ipu = get_prop.(
                        Ref(cl_h), Ref(cl_h_v), Ref(cl_h_u), (:ipl, :en, :ipr)
                    )
        pu = get_projector!(get_prop(cl_h, :pool_of_projectors), ipu, :CPU)
        pv = get_projector!(get_prop(cl_h, :pool_of_projectors), ipv, :CPU)
        @inbounds energies = en[pv[σ], pu]
    else
        energies = zeros(cluster_size(cl_h, cl_h_u))
    end
end

function cluster_size(clustered_hamiltonian::LabelledGraph{S, T}, vertex::T) where {S, T}
    length(get_prop(clustered_hamiltonian, vertex, :spectrum).energies)
end

function exact_cond_prob(clustered_hamiltonian::LabelledGraph{S, T}, beta, target_state::Dict) where {S, T}  # TODO: Not going to work without PoolOfProjectors
    ver = vertices(clustered_hamiltonian)
    rank = cluster_size.(Ref(clustered_hamiltonian), ver)
    states = [Dict(ver .=> σ) for σ ∈ Iterators.product([1:r for r ∈ rank]...)]
    energies = SpinGlassNetworks.energy.(Ref(clustered_hamiltonian), states)
    prob = exp.(-beta .* energies)
    prob ./= sum(prob)
    sum(prob[findall([all(s[k] == v for (k, v) ∈ target_state) for s ∈ states])])
 end

function truncate_clustered_hamiltonian(cl_h::LabelledGraph{S, T}, states::Dict) where {S, T}

    new_cl_h = LabelledGraph{MetaDiGraph}(vertices(cl_h))
    new_lp = PoolOfProjectors{Int}()

    for v ∈ vertices(new_cl_h)
        cl = get_prop(cl_h, v, :cluster)
        sp = get_prop(cl_h, v, :spectrum)
        if sp.states == Vector{Int64}[]
            sp = Spectrum(sp.energies[states[v]], sp.states, [])
        else
            sp = Spectrum(sp.energies[states[v]], sp.states[states[v]])
        end
        set_props!(new_cl_h, v, Dict(:cluster => cl, :spectrum => sp))
    end

    for e ∈ edges(cl_h)
        v, w = src(e), dst(e)
        add_edge!(new_cl_h, v, w)
        outer_edges = get_prop(cl_h, v, w, :outer_edges)
        ipl = get_prop(cl_h, v, w, :ipl)
        pl = get_projector!(get_prop(cl_h, :pool_of_projectors), ipl, :CPU)
        ipr = get_prop(cl_h, v, w, :ipr)
        pr = get_projector!(get_prop(cl_h, :pool_of_projectors), ipr, :CPU)
        en = get_prop(cl_h, v, w, :en)
        pl = pl[states[v]]
        pr = pr[states[w]]
        pl_transition, pl_unique = rank_reveal(pl, :PE)
        pr_transition, pr_unique = rank_reveal(pr, :PE)
        en = en[pl_unique, pr_unique]
        ipl = add_projector!(new_lp, pl_transition)
        ipr = add_projector!(new_lp, pr_transition)
        set_props!(
                  new_cl_h, v, w, Dict(:outer_edges => outer_edges, :ipl => ipl, :en => en, :ipr => ipr)
              )
    end
    set_props!(new_cl_h, Dict(:pool_of_projectors => new_lp))

    new_cl_h
end
