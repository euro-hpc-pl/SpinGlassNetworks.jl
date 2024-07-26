export truncate_potts_hamiltonian_2site_energy,
    truncate_potts_hamiltonian_1site_BP,
    truncate_potts_hamiltonian_2site_BP,
    select_numstate_best

"""
$(TYPEDSIGNATURES)

Truncates a Potts Hamiltonian using belief propagation (BP) for a single site cluster.

This function employs belief propagation (BP) to approximate the most probable states and energies for a Potts Hamiltonian
associated with a single-site cluster. It then truncates the Potts Hamiltonian based on the most probable states.

# Arguments:
- `potts_h::LabelledGraph{S, T}`: The Potts Hamiltonian represented as a labeled graph.
- `num_states::Int`: The maximum number of most probable states to keep.
- `beta::Real (optional)`: The inverse temperature parameter for the BP algorithm. Default is 1.0.
- `tol::Real (optional)`: The tolerance value for convergence in BP. Default is 1e-10.
- `iter::Int (optional)`: The maximum number of BP iterations. Default is 1.

# Returns:
- `LabelledGraph{S, T}`: A truncated Potts Hamiltonian.
"""
function truncate_potts_hamiltonian_1site_BP(
    potts_h::LabelledGraph{S,T},
    num_states::Int;
    beta = 1.0,
    tol = 1e-10,
    iter = 1,
) where {S,T}
    states = Dict()
    beliefs = belief_propagation(potts_h, beta; tol = tol, iter = iter)
    for node in vertices(potts_h)
        indices = partialsortperm(beliefs[node], 1:min(num_states, length(beliefs[node])))
        push!(states, node => indices)
    end
    truncate_potts_hamiltonian(potts_h, states)
end

"""
$(TYPEDSIGNATURES)

Truncate a Potts Hamiltonian based on 2-site energy states.

This function truncates a Potts Hamiltonian by considering 2-site energy states and selecting the most probable states 
to keep. It computes the energies for all 2-site combinations and selects the states that maximize the probability.

# Arguments:
- `potts_h::LabelledGraph{S, T}`: The Potts Hamiltonian represented as a labeled graph.
- `num_states::Int`: The maximum number of most probable states to keep.

# Returns:
- `LabelledGraph{S, T}`: A truncated Potts Hamiltonian.
"""
function truncate_potts_hamiltonian_2site_energy(
    potts_h::LabelledGraph{S,T},
    num_states::Int,
) where {S,T}
    # TODO: name to be clean to make it consistent with square2 and squarestar2
    states = Dict()
    for node in vertices(potts_h)
        if node in keys(states)
            continue
        end
        i, j, _ = node
        E1 = copy(get_prop(potts_h, (i, j, 1), :spectrum).energies)
        E2 = copy(get_prop(potts_h, (i, j, 2), :spectrum).energies)
        E = energy_2site(potts_h, i, j) .+ reshape(E1, :, 1) .+ reshape(E2, 1, :)
        sx, sy = size(E)
        E = reshape(E, sx * sy)
        ind1, ind2 = select_numstate_best(E, sx, num_states)
        push!(states, (i, j, 1) => ind1)
        push!(states, (i, j, 2) => ind2)
    end
    truncate_potts_hamiltonian(potts_h, states)
end

function load_file(filename)
    if isfile(filename)
        try
            load_object(string(filename))
        catch e
            return nothing
        end
    else
        return nothing
    end
end

"""
$(TYPEDSIGNATURES)

Truncate a Potts Hamiltonian based on 2-site belief propagation states.

This function truncates a Potts Hamiltonian by considering 2-site belief propagation states and selecting the most probable states 
to keep. It computes the beliefs for all 2-site combinations and selects the states that maximize the probability.

# Arguments:
- `potts_h::LabelledGraph{S, T}`: The Potts Hamiltonian represented as a labelled graph.
- `beliefs::Dict`: A dictionary containing belief values for 2-site interactions.
- `num_states::Int`: The maximum number of most probable states to keep.
- `beta::Real (optional)`: The inverse temperature parameter (default is 1.0).

# Returns:
- `LabelledGraph{S, T}`: A truncated Potts Hamiltonian.
"""
function truncate_potts_hamiltonian_2site_BP(
    potts_h::LabelledGraph{S,T},
    beliefs::Dict,
    num_states::Int,
    result_folder::String = "results_folder",
    inst::String = "inst";
    beta = 1.0,
) where {S,T}
    states = Dict()

    saved_states = load_file(joinpath(result_folder, "$(inst).jld2"))
    for node in vertices(potts_h)
        if node in keys(states)
            continue
        end
        i, j, _ = node
        sx =
            has_vertex(potts_h, (i, j, 1)) ?
            length(get_prop(potts_h, (i, j, 1), :spectrum).energies) : 1
        E = beliefs[(i, j)]
        ind1, ind2 = select_numstate_best(E, sx, num_states)
        push!(states, (i, j, 1) => ind1)
        push!(states, (i, j, 2) => ind2)
    end
    path = joinpath(result_folder, "$(inst).jld2")
    save_object(string(path), states)
    truncate_potts_hamiltonian(potts_h, states)
end

"""
$(TYPEDSIGNATURES)

Select a specified number of best states based on energy.

This function selects a specified number of best states from a list of energies based on energy values in two nodes of Potts Hamiltonian. 
It fine-tunes the selection to ensure that the resulting states have the expected number.

# Arguments:
- `E::Vector{Real}`: A vector of energy values.
- `sx::Int`: The size of the Potts Hamiltonian for one of the nodes.
- `num_states::Int`: The desired number of states to select.

# Returns:
- `Tuple{Vector{Int}, Vector{Int}}`: A tuple containing two vectors of indices, `ind1` and `ind2`, 
which represent the selected states for two nodes of a Potts Hamiltonian.
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

function truncate_potts_hamiltonian(
    potts_h,
    β,
    cs,
    result_folder,
    inst;
    tol = 1e-6,
    iter = iter,
)
    states = Dict()
    saved_states = load_file(joinpath(result_folder, "$(inst).jld2"))
    if isnothing(saved_states)
        new_potts_h = potts_hamiltonian_2site(potts_h, β)
        beliefs = belief_propagation(new_potts_h, β; tol = 1e-6, iter = iter)
        potts_h = truncate_potts_hamiltonian_2site_BP(
            potts_h,
            beliefs,
            cs,
            result_folder,
            inst;
            beta = β,
        )
    else
        states = saved_states
        potts_h = truncate_potts_hamiltonian(potts_h, states)
    end
    potts_h
end
