export
    truncate_factor_graph_2site_energy,
    truncate_factor_graph_1site_BP,
    truncate_factor_graph_2site_BP


function truncate_factor_graph_1site_BP(fg::LabelledGraph{S, T}, num_states::Int; beta=1.0, tol=1e-10, iter=1) where {S, T}
    states = Dict()
    beliefs = belief_propagation(fg, beta; tol=tol, iter=iter)
    for node in vertices(fg)
        indices = partialsortperm(beliefs[node], 1:min(num_states, length(beliefs[node])))
        push!(states, node => indices)
    end
    truncate_factor_graph(fg, states)
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
        sx = has_vertex(fg, (i, j, 1)) ? length(get_prop(fg, (i, j, 1), :spectrum).energies) : 1
        E = beliefs[(i, j)]
        ind1, ind2 = select_numstate_best(E, sx, num_states)
        push!(states, (i, j, 1) => ind1)
        push!(states, (i, j, 2) => ind2)
    end
    truncate_factor_graph(fg, states)
end

function select_numstate_best(E, sx, num_states)
    # truncate based on energy in two nodes of factor graph;
    # resulting states are a product of states in two nodes, so we have to fine-tune to end up with expected number of states

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
