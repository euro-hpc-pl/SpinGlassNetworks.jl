export
    factor_graph,
    rank_reveal,
    projectors,
    split_into_clusters,
    decode_factor_graph_state,
    energy,
    cluster_size,
    truncate_factor_graph_2site_energy,
    truncate_factor_graph_1site_BP,
    truncate_factor_graph_2site_BP,
    belief_propagation,
    belief_propagation_old,
    exact_cond_prob,
    beliefs_2site,
    factor_graph_2site
"""
Groups spins into clusters: Dict(factor graph coordinates -> group of spins in Ising graph)
"""
function split_into_clusters(ig::LabelledGraph{S, T}, assignment_rule) where {S, T}
    cluster_id_to_verts = Dict(i => T[] for i in values(assignment_rule))
    for v in vertices(ig) push!(cluster_id_to_verts[assignment_rule[v]], v) end
    Dict(i => first(cluster(ig, verts)) for (i, verts) ∈ cluster_id_to_verts)
end

"""
Create factor graph.
Factor graph order introduced as a natural order in factor graph coordinates.
"""
function factor_graph(
    ig::IsingGraph,
    num_states_cl::Int;
    spectrum::Function=full_spectrum,
    cluster_assignment_rule::Dict{Int, T} # e.g. square lattice
) where T
    ns = Dict(i => num_states_cl for i ∈ Set(values(cluster_assignment_rule)))
    factor_graph(ig, ns, spectrum=spectrum, cluster_assignment_rule=cluster_assignment_rule)
end

function factor_graph(
    ig::IsingGraph,
    num_states_cl::Dict{T, Int};
    spectrum::Function=full_spectrum,
    cluster_assignment_rule::Dict{Int, T}
) where T
    L = maximum(values(cluster_assignment_rule))
    fg = LabelledGraph{MetaDiGraph}(sort(unique(values(cluster_assignment_rule))))

    for (v, cl) ∈ split_into_clusters(ig, cluster_assignment_rule)
        sp = spectrum(cl, num_states=get(num_states_cl, v, basis_size(cl)))
        set_props!(fg, v, Dict(:cluster => cl, :spectrum => sp))
    end

    for (i, v) ∈ enumerate(vertices(fg)), w ∈ vertices(fg)[i+1:end]
        cl1, cl2 = get_prop(fg, v, :cluster), get_prop(fg, w, :cluster)
        outer_edges, J = inter_cluster_edges(ig, cl1, cl2)

        if !isempty(outer_edges)
            ind1 = any(i -> i != 0, J, dims=2)
            ind2 = any(i -> i != 0, J, dims=1)
            ind1 = reshape(ind1, length(ind1))
            ind2 = reshape(ind2, length(ind2))
            JJ = J[ind1, ind2]

            states_v = get_prop(fg, v, :spectrum).states
            states_w = get_prop(fg, w, :spectrum).states

            pl, unique_states_v = rank_reveal([s[ind1] for s ∈ states_v], :PE)
            pr, unique_states_w = rank_reveal([s[ind2] for s ∈ states_w], :PE)
            en = inter_cluster_energy(unique_states_v, JJ, unique_states_w)

            add_edge!(fg, v, w)
            set_props!(
                fg, v, w, Dict(:outer_edges => outer_edges, :pl => pl, :en => en, :pr => pr)
            )
        end
    end
    fg
end

function factor_graph(
    ig::IsingGraph; spectrum::Function=full_spectrum, cluster_assignment_rule::Dict{Int, T}
) where T
    factor_graph(
      ig, Dict{T, Int}(), spectrum=spectrum, cluster_assignment_rule=cluster_assignment_rule
    )
end

function rank_reveal(energy, order=:PE)
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
function decode_factor_graph_state(fg, state::Vector{Int})
    ret = Dict{Int, Int}()
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

"""
TODO: write it better (for now this is only for testing).
"""
function energy(ig::IsingGraph, ig_state::Dict{Int, Int})
    en = 0.0
    for (i, σ) ∈ ig_state
        en += get_prop(ig, i, :h) * σ
        for (j, η) ∈ ig_state
            if has_edge(ig, i, j)
                en += σ * get_prop(ig, i, j, :J) * η / 2.0
            elseif has_edge(ig, j, i)
                en += σ * get_prop(ig, j, i, :J) * η / 2.0
            end
        end
    end
    en
end

function energy(fg::LabelledGraph{S, T}, σ::Dict{T, Int}) where {S, T}
    en_fg = 0.0
    for v ∈ vertices(fg) en_fg += get_prop(fg, v, :spectrum).energies[σ[v]] end
    for edge ∈ edges(fg)
        pl, pr = get_prop(fg, edge, :pl), get_prop(fg, edge, :pr)
        en = get_prop(fg, edge, :en)
        en_fg += en[pl[σ[src(edge)]], pr[σ[dst(edge)]]]
    end
    en_fg
end

function cluster_size(factor_graph::LabelledGraph{S, T}, vertex::T) where {S, T}
    length(get_prop(factor_graph, vertex, :spectrum).energies)
end


function truncate_factor_graph_2site_energy(fg::LabelledGraph{S, T}, num_states::Int) where {S, T}  # TODO: name to be clean to make it consistent with square2 and squarestar2
    states = Dict()
    for node in vertices(fg)
        if node in keys(states) continue end
        i, j, _ = node
        E1 = copy(get_prop(fg, (i, j, 1), :spectrum).energies)
        E2 = copy(get_prop(fg, (i, j, 2), :spectrum).energies)
        E = energy_2site(fg, i, j) .+ reshape(E1, :, 1) .+ reshape(E2, 1, :)
        sx, sy = size(E)
        E = reshape(E, sx * sy)
        ind1, ind2 = select_numstate_best(E, sx, num_states)
        push!(states, (i, j, 1) => ind1)
        push!(states, (i, j, 2) => ind2)
    end
    truncate_factor_graph(fg, states)
end

function truncate_factor_graph_2site_BP(fg::LabelledGraph{S, T}, num_states::Int; beta=1.0, tol=1e-6, iter=1) where {S, T}  # TODO: name to be clean to make it consistent with square2 and squarestar2
    new_fg = factor_graph_2site(fg, beta)
    beliefs = belief_propagation(new_fg, beta; tol, iter)
    states = Dict()
    for node in vertices(fg)
        if node in keys(states) continue end
        i, j, _ = node
        sx = length(get_prop(fg, (i, j, 1), :spectrum).energies)
        E = beliefs[(i,j)]
        ind1, ind2 = select_numstate_best(E, sx, num_states)
        push!(states, (i, j, 1) => ind1)
        push!(states, (i, j, 2) => ind2)
    end
    truncate_factor_graph(fg, states)
end

# truncate based on energy in two nodes of factor graph; resulting states are a product of states in two nodes, so we have to fine-tune to end up with expected number of states
function select_numstate_best(E, sx, num_states)

    low, high = 1, min(num_states, length(E))

    while true
        guess = div(low + high, 2)
        ind = partialsortperm(E, 1:guess)
        ind1 = mod.(ind .- 1, sx) .+ 1
        ind2 = div.(ind .- 1, sx) .+ 1
        ind1 = sort([Set(ind1)...])
        ind2 = sort([Set(ind2)...])
        if high - low <= 1
            return ind1, ind2
        end
        if length(ind1) * length(ind2) > num_states
            high = guess
        else
            low = guess
        end
    end
end

function get_neighbors(graph::LabelledGraph{S, T}, vertex::NTuple) where {S, T}
    neighbors = []
    for edge in edges(graph)
        src_node, dst_node = src(edge), dst(edge)
        if src_node == vertex
            push!(neighbors, dst_node)
        elseif dst_node == vertex
            push!(neighbors, src_node)
        end
    end
    return neighbors
end

# Works for Pegasus and Zephyr
function truncate_factor_graph(fg::LabelledGraph{S, T}, states::Dict) where {S, T}

    new_fg = LabelledGraph{MetaDiGraph}(vertices(fg))
    
    for v ∈ vertices(new_fg)
        cl = get_prop(fg, v, :cluster)
        sp = get_prop(fg, v, :spectrum)
        sp = Spectrum(sp.energies[states[v]], sp.states[states[v]])
        set_props!(new_fg, v, Dict(:cluster => cl, :spectrum => sp))
    end

    for e ∈ edges(fg)
        v, w = src(e), dst(e)
        add_edge!(new_fg, v, w)
        outer_edges = get_prop(fg, v, w, :outer_edges)
        pl = get_prop(fg, v, w, :pl)
        pr = get_prop(fg, v, w, :pr)
        en = get_prop(fg, v, w, :en)
        pl = pl[states[v]]
        pr = pr[states[w]]
        pl_transition, pl_unique = rank_reveal(pl, :PE)
        pr_transition, pr_unique = rank_reveal(pr, :PE)
        en = en[pl_unique, pr_unique]

        set_props!(
                  new_fg, v, w, Dict(:outer_edges => outer_edges, :pl => pl_transition, :en => en, :pr => pr_transition)
              )
    end
    new_fg
end

function truncate_factor_graph_1site_BP(fg::LabelledGraph{S, T}, num_states::Int; beta=1.0, tol=1e-6, iter=1) where {S, T}
    states = Dict()
    beliefs = belief_propagation(fg, beta; tol=tol, iter=iter)
    for node in vertices(fg)
        indices = partialsortperm(beliefs[node], 1:min(num_states, length(beliefs[node])))
        push!(states, node => indices)
    end
    truncate_factor_graph(fg, states)
end

# function truncate_factor_graph_2site_BP(fg::LabelledGraph{S, T}, num_states::Int; beta=1.0, tol=1e-6, iter=1) where {S, T}
#     states = Dict()
#     beliefs, messages_av = belief_propagation(fg, beta; tol=tol, iter=iter, output_message=true)

#     for node in vertices(fg)
#         if node in keys(states) continue end
#         i, j, _ = node
#         temp = beliefs_2site(fg, i, j, messages_av, beta)
#         sx, sy = size(temp)
#         temp = reshape(temp, sx * sy)
#         ind1, ind2 = select_numstate_best(temp, sx, num_states)
#         push!(states, (i, j, 1) => ind1)
#         push!(states, (i, j, 2) => ind2)
#     end
    
#     truncate_factor_graph(fg, states)
# end

function beliefs_2site(fg::LabelledGraph{S, T}, i::Int, j::Int, messages_av::Dict, beta::Real) where {S, T}
    temp = energy_2site(fg, i, j)
    temp = exp.(-beta*temp)
    if ((i, j, 1), (i, j, 2)) in keys(messages_av)
        temp = temp .* reshape(messages_av[(i, j, 1), (i, j, 2)], :, 1) 
        temp = temp .*reshape(messages_av[(i, j, 2), (i, j, 1)], 1, :)
    end
    temp ./= sum(temp)
end

function energy_2site(fg::LabelledGraph{S, T}, i::Int, j::Int) where {S, T}
    if has_edge(fg, (i, j, 1), (i, j, 2))
        en12 = copy(get_prop(fg, (i, j, 1), (i, j, 2), :en))
        pl = copy(get_prop(fg, (i, j, 1), (i, j, 2), :pl))
        pr = copy(get_prop(fg, (i, j, 1), (i, j, 2), :pr))
        int_eng = en12[pl, pr]
    elseif has_edge(fg, (i, j, 2), (i, j, 1))
        en21 = copy(get_prop(fg, (i, j, 2), (i, j, 1), :en))
        pl = copy(get_prop(fg, (i, j, 2), (i, j, 1), :pl))
        pr = copy(get_prop(fg, (i, j, 2), (i, j, 1), :pr))
        int_eng = en21[pl, pr]'
    else
        int_eng = zeros(1, 1)
    end
    int_eng
end

# Works for Pegasus and Zephyr
# function truncate_factor_graph_1site(fg::LabelledGraph{S, T}, beta::Real, num_states::Int; tol=1e-6, iter=1) where {S, T}
#     states = Dict()
#     beliefs = belief_propagation(fg, beta; tol=tol, iter=iter)

#     # Truncate the state space based on the belief probabilities
#     for node in vertices(fg)
#         indices = partialsortperm(beliefs[node], 1:min(num_states, length(beliefs[node])), rev=true)
#         push!(states, node => indices)
#     end

#     # Create a new factor graph with truncated state space
#     new_fg = LabelledGraph{MetaDiGraph}(vertices(fg))
    
#     for v ∈ vertices(new_fg)
#         cl = get_prop(fg, v, :cluster)
#         sp = get_prop(fg, v, :spectrum)
#         sp = Spectrum(sp.energies[states[v]], sp.states[states[v]])
#         set_props!(new_fg, v, Dict(:cluster => cl, :spectrum => sp))
#     end

#     for e ∈ edges(fg)
#         v, w = src(e), dst(e)
#         add_edge!(new_fg, v, w)
#         outer_edges = get_prop(fg, v, w, :outer_edges)
#         pl = get_prop(fg, v, w, :pl)
#         pr = get_prop(fg, v, w, :pr)
#         en = get_prop(fg, v, w, :en)
#         pl = pl[states[v]]
#         pr = pr[states[w]]
#         pl_transition, pl_unique = rank_reveal(pl, :PE)
#         pr_transition, pr_unique = rank_reveal(pr, :PE)
#         en = en[pl_unique, pr_unique]

#         set_props!(
#                   new_fg, v, w, Dict(:outer_edges => outer_edges, :pl => pl_transition, :en => en, :pr => pr_transition)
#               )
#     end
#     new_fg
# end


function belief_propagation_old(fg, beta; tol=1e-6, iter=1, output_message=false)
    messages_va = Dict()
    messages_av = Dict()
    beliefs = Dict()

    # Initialize messages with uniform probabilities
    for v in vertices(fg)
        ns = length(get_prop(fg, v, :spectrum).states)
        push!(beliefs, v => ones(ns)./ns)
        for neighbor in get_neighbors(fg, v)
            push!(messages_va, (v, neighbor) => ones(ns)./ns)
            push!(messages_av, (neighbor, v) => ones(ns)./ns)
        end
    end 

    # Perform message passing until convergence
    converged = false
    iteration = 0
    while !converged && iteration < iter  # Set an appropriate number of iterations and convergence threshold
        iteration += 1
        old_beliefs = deepcopy(beliefs)
        for v in vertices(fg)
            for neighbor in get_neighbors(fg, v)
                #update messages from vertex to edge
                messages_va[v, neighbor] = beliefs[v]./messages_av[neighbor, v]
                messages_va[v, neighbor] ./= sum(messages_va[v, neighbor])
            end
        end
        for v in vertices(fg)
            for neighbor in get_neighbors(fg, v)
                #update messages from edge to vertex
                if has_edge(fg, v, neighbor)
                    E_bond = get_prop(fg, v, neighbor, :en)
                    pl, pr = get_prop(fg, v, neighbor, :pl), get_prop(fg, v, neighbor, :pr)
                elseif has_edge(fg, neighbor, v)
                    E_bond = get_prop(fg, neighbor, v, :en)'
                    pl, pr = get_prop(fg, neighbor, v, :pr), get_prop(fg, neighbor, v, :pl)
                end
                temp = zeros(maximum(pr))
                for (i, r) in enumerate(pr)
                    temp[r] += messages_va[neighbor, v][i]
                end
                temp = exp.(-beta * E_bond) * temp
                messages_av[neighbor, v] = temp[pl]
                messages_av[neighbor, v] ./= sum(messages_av[neighbor, v])
            end
        end
        
        for v in vertices(fg)
            E_local = get_prop(fg, v, :spectrum).energies
            beliefs[v] = exp.(-E_local * beta)
            for neighbor in get_neighbors(fg, v)
                #update beliefs
                beliefs[v] .*= messages_av[neighbor, v]
            end
            # Normalize the beliefs
            beliefs[v] ./= sum(beliefs[v])
        end

        # Check convergence
        converged = all([all(abs.(old_beliefs[v] .- beliefs[v]) .< tol) for v in keys(beliefs)])
    end

    output_message ? (beliefs, messages_av) : beliefs
end

function exact_cond_prob(factor_graph::LabelledGraph{S, T}, beta, target_state::Dict) where {S, T}  # TODO: Not going to work without PoolOfProjectors
    ver = vertices(factor_graph)
    rank = cluster_size.(Ref(factor_graph), ver)
    states = [Dict(ver .=> σ) for σ ∈ Iterators.product([1:r for r ∈ rank]...)]
    energies = SpinGlassNetworks.energy.(Ref(factor_graph), states)
    prob = exp.(-beta .* energies)
    prob ./= sum(prob)
    sum(prob[findall([all(s[k] == v for (k, v) ∈ target_state) for s ∈ states])])
 end

 function belief_propagation(fg, beta; tol=1e-6, iter=1, output_message=false)
    messages_va = Dict()
    messages_av = Dict()
    beliefs = Dict()

    # Initialize messages with uniform probabilities
    for v in vertices(fg)
        ns = length(get_prop(fg, v, :spectrum).energies)
        push!(beliefs, v => ones(ns)./ns)
        for neighbor in get_neighbors(fg, v)
            pr = has_edge(fg, v, neighbor) ? get_prop(fg, v, neighbor, :pr) : get_prop(fg, neighbor, v, :pl)
            pl = has_edge(fg, neighbor, v) ? get_prop(fg, neighbor, v, :pr) : get_prop(fg, v, neighbor, :pl)
            temp = zeros(maximum(pr))
            for (i, r) in enumerate(pr)
                temp[r] += 1
            end
            push!(messages_va, (v, neighbor) => temp ./ sum(temp))
            push!(messages_av, (neighbor, v) => ones(maximum(pl)))
        end
    end 

    # Perform message passing until convergence
    converged = false
    iteration = 0
    while !converged && iteration < iter  # Set an appropriate number of iterations and convergence threshold
        iteration += 1
        old_beliefs = deepcopy(beliefs)
        for v in vertices(fg)
            for neighbor in get_neighbors(fg, v)
                #update messages from vertex to edge
                E_local = get_prop(fg, v, :spectrum).energies
                E_local = E_local .- minimum(E_local)
                messages_va[v, neighbor] = exp.(-E_local * beta)
                for n in get_neighbors(fg, v)
                    if n != neighbor
                        pl = has_edge(fg, n, v) ? get_prop(fg, n, v, :pr) : get_prop(fg, v, n, :pl)
                        messages_va[v, neighbor] .*= messages_av[n, v][pl]
                    end
                end
                messages_va[v, neighbor] ./= sum(messages_va[v, neighbor])
                pr = has_edge(fg, v, neighbor) ? get_prop(fg, v, neighbor, :pl) : get_prop(fg, neighbor, v, :pr)
                temp = zeros(maximum(pr))
                for (i, r) in enumerate(pr)
                    temp[r] += messages_va[v, neighbor][i]
                end
                messages_va[v, neighbor] = temp
            end
        end
        for v in vertices(fg)
            for neighbor in get_neighbors(fg, v)
                #update messages from edge to verte
                # messages_av[neighbor, v] = exp.(-beta * E_bond) * messages_va[neighbor, v]
                E_bond = has_edge(fg, v, neighbor) ? get_prop(fg, v, neighbor, :en) : get_prop(fg, neighbor, v, :en)'
                messages_av[neighbor, v] = update_message(messages_va[neighbor, v], E_bond, v, neighbor, beta)
            end
        end
        
        for v in vertices(fg)
            E_local = get_prop(fg, v, :spectrum).energies
            beliefs[v] = exp.(-E_local * beta)
            for neighbor in get_neighbors(fg, v)
                #update beliefs
                pl = has_edge(fg, neighbor, v) ? get_prop(fg, neighbor, v, :pr) : get_prop(fg, v, neighbor, :pl)
                beliefs[v] .*= messages_av[neighbor, v][pl]
            end
            # Normalize the beliefs
            beliefs[v] = -log.(beliefs[v])./beta
            beliefs[v] = beliefs[v] .- minimum(beliefs[v])
        end

        # Check convergence
        converged = all([all(abs.(old_beliefs[v] .- beliefs[v]) .< tol) for v in keys(beliefs)])
    end

    output_message ? (beliefs, messages_av) : beliefs
end

struct Energy{T <: Real} 
    e11::AbstractMatrix{T}
    e12::AbstractMatrix{T}
    e21::AbstractMatrix{T}
    e22::AbstractMatrix{T}
end

Base.adjoint(s::Energy) = Energy(s.e11', s.e21', s.e12', s.e22')

function update_message(message::Vector, E_bond, v, neighbor, beta)
    E_bond = E_bond .- minimum(E_bond)
    exp.(-beta * E_bond) * message
end

function update_message(message::Vector, E_bond::Energy, v, neighbor, beta)
    e11, e12, e21, e22 = E_bond.e11, E_bond.e12, E_bond.e21, E_bond.e22
    @cast E[(l1, l2), (r1, r2)] := e11[l1, r1] + e21[l2, r1] + e12[l1, r2] + e22[l2, r2]
    exp.(-beta * E) * message
end

function unify_vertices(vertices::Vector{Tuple{Int64, Int64, Int64}})
    unified_vertices = Vector{Tuple{Int64, Int64}}()
    for vertex in vertices
        push!(unified_vertices, vertex[1:2])
    end
    return unique(unified_vertices)
end

function factor_graph_2site(fg::LabelledGraph{S, T}, beta::Real) where {S, T}
    new_fg = LabelledGraph{MetaDiGraph}(unify_vertices(vertices(fg)))

    for v in vertices(fg)
        i, j, _ = v
        E1 = copy(get_prop(fg, (i, j, 1), :spectrum).energies)
        E2 = copy(get_prop(fg, (i, j, 2), :spectrum).energies)
        E = energy_2site(fg, i, j) .+ reshape(E1, :, 1) .+ reshape(E2, 1, :)
        sp = Spectrum(reshape(E, :), [])
        set_props!(new_fg, (i, j), Dict(:spectrum => sp))
    end

    edge_states = Set()
    for e ∈ edges(fg)
        if e in edge_states continue end
        v, w = src(e), dst(e)
        v1, v2, _ = v
        w1, w2, _ = w
        if (v1, v2) != (w1, w2) 
            add_edge!(new_fg, (v1, v2), (w1, w2))
      
            E, pl, pr = merge_vertices(fg, beta, v, w)
            push!(edge_states, sort([(v1, v2), (w1, w2)]))

            set_props!(
                  new_fg, (v1, v2), (w1, w2), Dict(:pl => pl, :en => E, :pr => pr)
              )
        end
    end
    new_fg
end

function merge_vertices(fg::LabelledGraph{S, T}, β::Real, node1::NTuple{3, Int64}, node2::NTuple{3, Int64}
    ) where {S, T}
    i1, j1, k1 = node1
    i2, j2, k2 = node2

    p21l = projector(fg, (i1, j1, 2), (i2, j2, 1))
    p22l = projector(fg, (i1, j1, 2), (i2, j2, 2))
    p12l = projector(fg, (i1, j1, 1), (i2, j2, 2))
    p11l = projector(fg, (i1, j1, 1), (i2, j2, 1))

    p1l, (p11l, p12l) = fuse_projectors((p11l, p12l))
    p2l, (p21l, p22l) = fuse_projectors((p21l, p22l))

    p11r = projector(fg, (i2, j2, 1), (i1, j1, 1))
    p21r = projector(fg, (i2, j2, 1), (i1, j1, 2))
    p12r = projector(fg, (i2, j2, 2), (i1, j1, 1))
    p22r = projector(fg, (i2, j2, 2), (i1, j1, 2))

    p1r, (p11r, p21r) = fuse_projectors((p11r, p21r))
    p2r, (p12r, p22r) = fuse_projectors((p12r, p22r))

    pl = outer_projector(p1l, p2l)
    pr = outer_projector(p1r, p2r)

    e11 = interaction_energy(fg, (i1, j1, 1), (i2, j2, 1))
    e12 = interaction_energy(fg, (i1, j1, 1), (i2, j2, 2))
    e21 = interaction_energy(fg, (i1, j1, 2), (i2, j2, 1))
    e22 = interaction_energy(fg, (i1, j1, 2), (i2, j2, 2))

    e11 = e11[p11l, p11r]
    e21 = e21[p21l, p21r]
    e12 = e12[p12l, p12r]
    e22 = e22[p22l, p22r]

    Energy(e11, e12, e21, e22), pl, pr
end

function interaction_energy(fg::LabelledGraph{S, T}, v::NTuple{3, Int64}, w::NTuple{3, Int64}) where {S, T}
    if has_edge(fg, w, v)
        get_prop(fg, w, v, :en)'
    elseif has_edge(fg, v, w)
        get_prop(fg, v, w, :en)
    else
        zeros(1, 1)
    end
end

function projector(fg::LabelledGraph{S, T}, v::NTuple{3, Int64}, w::NTuple{3, Int64}) where {S, T}
    if has_edge(fg, w, v)
        p = get_prop(fg, w, v, :pr)
    elseif has_edge(fg, v, w)
        p = get_prop(fg, v, w, :pl)
    else
        p = ones(Int, v ∈ vertices(fg) ? length(get_prop(fg, v, :spectrum).energies) : 1)
    end
end

function fuse_projectors(
    projectors::NTuple{N, K}
    ) where {N, K}
    fused, transitions_matrix = rank_reveal(hcat(projectors...), :PE)
    transitions = Tuple(Array(t) for t ∈ eachcol(transitions_matrix))
    fused, transitions
end

function outer_projector(p1::Array{T, 1}, p2::Array{T, 1}) where T <: Number
    reshape(reshape(p1, :, 1) .+ maximum(p1) .* reshape(p2 .- 1, 1, :), :)
end