export
    truncate_clustered_hamiltonian_2site_energy,
    truncate_clustered_hamiltonian_1site_BP,
    truncate_clustered_hamiltonian_2site_BP,
    select_numstate_best

"""
$(TYPEDSIGNATURES)

Truncates a clustered Hamiltonian using belief propagation (BP) for a single site cluster.

This function employs belief propagation (BP) to approximate the most probable states and energies for a clustered Hamiltonian
associated with a single-site cluster. It then truncates the clustered Hamiltonian based on the most probable states.

# Arguments:
- `cl_h::LabelledGraph{S, T}`: The clustered Hamiltonian represented as a labeled graph.
- `num_states::Int`: The maximum number of most probable states to keep.
- `beta::Real (optional)`: The inverse temperature parameter for the BP algorithm. Default is 1.0.
- `tol::Real (optional)`: The tolerance value for convergence in BP. Default is 1e-10.
- `iter::Int (optional)`: The maximum number of BP iterations. Default is 1.

# Returns:
- `LabelledGraph{S, T}`: A truncated clustered Hamiltonian.
"""
function truncate_clustered_hamiltonian_1site_BP(
    cl_h::LabelledGraph{S, T}, 
    num_states::Int; 
    beta=1.0, 
    tol=1e-10, 
    iter=1
    ) where {S, T}
    states = Dict()
    beliefs = belief_propagation(cl_h, beta; tol=tol, iter=iter)
    for node in vertices(cl_h)
        indices = partialsortperm(beliefs[node], 1:min(num_states, length(beliefs[node])))
        push!(states, node => indices)
    end
    truncate_clustered_hamiltonian(cl_h, states)
end

"""
$(TYPEDSIGNATURES)

Truncate a clustered Hamiltonian based on 2-site energy states.

This function truncates a clustered Hamiltonian by considering 2-site energy states and selecting the most probable states 
to keep. It computes the energies for all 2-site combinations and selects the states that maximize the probability.

# Arguments:
- `cl_h::LabelledGraph{S, T}`: The clustered Hamiltonian represented as a labeled graph.
- `num_states::Int`: The maximum number of most probable states to keep.

# Returns:
- `LabelledGraph{S, T}`: A truncated clustered Hamiltonian.
"""
function truncate_clustered_hamiltonian_2site_energy(cl_h::LabelledGraph{S, T}, num_states::Int) where {S, T}
    # TODO: name to be clean to make it consistent with square2 and squarestar2
    states = Dict()
    for node in vertices(cl_h)
        if node in keys(states) continue end
        i, j, _ = node
        E1 = copy(get_prop(cl_h, (i, j, 1), :spectrum).energies)
        E2 = copy(get_prop(cl_h, (i, j, 2), :spectrum).energies)
        E = energy_2site(cl_h, i, j) .+ reshape(E1, :, 1) .+ reshape(E2, 1, :)
        sx, sy = size(E)
        E = reshape(E, sx * sy)
        ind1, ind2 = select_numstate_best(E, sx, num_states)
        push!(states, (i, j, 1) => ind1)
        push!(states, (i, j, 2) => ind2)
    end
    truncate_clustered_hamiltonian(cl_h, states)
end

"""
$(TYPEDSIGNATURES)

Truncate a clustered Hamiltonian based on 2-site belief propagation states.

This function truncates a clustered Hamiltonian by considering 2-site belief propagation states and selecting the most probable states 
to keep. It computes the beliefs for all 2-site combinations and selects the states that maximize the probability.

# Arguments:
- `cl_h::LabelledGraph{S, T}`: The clustered Hamiltonian represented as a labeled graph.
- `beliefs::Dict`: A dictionary containing belief values for 2-site interactions.
- `num_states::Int`: The maximum number of most probable states to keep.
- `beta::Real (optional)`: The inverse temperature parameter (default is 1.0).

# Returns:
- `LabelledGraph{S, T}`: A truncated clustered Hamiltonian.
"""
function truncate_clustered_hamiltonian_2site_BP(
    cl_h::LabelledGraph{S, T}, 
    beliefs::Dict, 
    num_states::Int; 
    beta=1.0
    ) where {S, T}
    # TODO: name to be clean to make it consistent with square2 and squarestar2
    states = Dict()
    for node in vertices(cl_h)
        if node in keys(states) continue end
        i, j, _ = node
        sx = has_vertex(cl_h, (i, j, 1)) ? length(get_prop(cl_h, (i, j, 1), :spectrum).energies) : 1
        E = beliefs[(i, j)]
        ind1, ind2 = select_numstate_best(E, sx, num_states)
        push!(states, (i, j, 1) => ind1)
        push!(states, (i, j, 2) => ind2)
    end
    truncate_clustered_hamiltonian(cl_h, states)
end

"""
$(TYPEDSIGNATURES)

Select a specified number of best states based on energy.

This function selects a specified number of best states from a list of energies based on energy values in two nodes of clustered hamiltonian. It fine-tunes the selection to ensure that the resulting states have the expected number.

# Arguments:
- `E::Vector{Real}`: A vector of energy values.
- `sx::Int`: The size of the clustered Hamiltonian for one of the nodes.
- `num_states::Int`: The desired number of states to select.

# Returns:
- `Tuple{Vector{Int}, Vector{Int}}`: A tuple containing two vectors of indices, `ind1` and `ind2`, which represent the selected states for two nodes of a clustered Hamiltonian.
"""
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
