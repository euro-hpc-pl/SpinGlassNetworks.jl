
export
    belief_propagation,
    factor_graph_2site


function belief_propagation(fg, beta; tol=1e-6, iter=1)
    messages_ve = Dict()
    messages_ev = Dict()

    # Initialize messages with uniform probabilities
    for v in vertices(fg)
        for (n, pv, _) in get_neighbors(fg, v)
            push!(messages_ev, (n, v) => ones(maximum(pv)))
        end
    end

    # Perform message passing until convergence
    converged = false
    iteration = 0
    while !converged && iteration < iter  # Set an appropriate number of iterations and convergence threshold
        iteration += 1
        old_messages_ev = deepcopy(messages_ev)
        for v in vertices(fg)
            #update messages from vertex to edge
            for (n1, pv1, _) ∈ get_neighbors(fg, v)
                E_local = get_prop(fg, v, :spectrum).energies
                messages_ve[v, n1] = exp.(-(E_local .- minimum(E_local)) * beta)
                for (n2, pv2, _) in get_neighbors(fg, v)
                    if n1 == n2 continue end
                    messages_ve[v, n1] .*= messages_ev[n2, v][pv2]
                end
                messages_ve[v, n1] ./= sum(messages_ve[v, n1])
                temp = zeros(maximum(pv1))
                for (i, r) in enumerate(pv1)
                    temp[r] += messages_ve[v, n1][i]
                end
                messages_ve[v, n1] = temp
            end
        end

        #update messages from edge to verte
        for v in vertices(fg)
            for (n, _, en) ∈ get_neighbors(fg, v)
                messages_ev[n, v] = update_message(en, messages_ve[n, v], beta)
            end
        end

        # Check convergence
        converged = all([all(abs.(old_messages_ev[v] .- messages_ev[v]) .< tol) for v in keys(messages_ev)])
    end

    beliefs = Dict()
    for v in vertices(fg)
        E_local = get_prop(fg, v, :spectrum).energies
        beliefs[v] = exp.(-E_local * beta)
        for (n, pv, _) ∈ get_neighbors(fg, v)
            beliefs[v] .*= messages_ev[n, v][pv]
        end
        beliefs[v] = -log.(beliefs[v])./beta
        beliefs[v] = beliefs[v] .- minimum(beliefs[v])
    end

    beliefs
end

function get_neighbors(fg::LabelledGraph{S, T}, vertex::NTuple) where {S, T}
    neighbors = []
    for edge in edges(fg)
        src_node, dst_node = src(edge), dst(edge)
        if src_node == vertex
            en = get_prop(fg, src_node, dst_node, :en)
            pv = get_prop(fg, src_node, dst_node, :pl)
            push!(neighbors, (dst_node, pv, en))
        elseif dst_node == vertex
            en = get_prop(fg, src_node, dst_node, :en)'
            pv = get_prop(fg, src_node, dst_node, :pr)
            push!(neighbors, (src_node, pv, en))
        end
    end
    return neighbors
end

struct MergedEnergy{T <: Real}
    e11::AbstractMatrix{T}
    e12::AbstractMatrix{T}
    e21::AbstractMatrix{T}
    e22::AbstractMatrix{T}
end

Base.adjoint(s::MergedEnergy) = MergedEnergy(s.e11', s.e21', s.e12', s.e22')

function update_message( E_bond::AbstractArray, message::Vector, beta::Real)
    E_bond = E_bond .- minimum(E_bond)
    exp.(-beta * E_bond) * message
end

function update_message(E_bond::MergedEnergy, message::Vector, beta::Real)
    #  equivalent to
    e11, e12, e21, e22 = E_bond.e11, E_bond.e12, E_bond.e21, E_bond.e22
    @cast E[(l1, l2), (r1, r2)] := e11[l1, r1] + e21[l2, r1] + e12[l1, r2] + e22[l2, r2]
    exp.(-beta * E) * message

    # but without creating large matrix E if not optimal
    # e11, e12, e21, e22 = E_bond.e11', E_bond.e21', E_bond.e12', E_bond.e22'
    # sbt = length(message)
    # sl1, sl2, sr1, sr2 = size(e11, 1), size(e22, 1), size(e11, 2), size(e22, 2)
    # sinter = sbt * max(sl1 * sl2 * min(sr1, sr2), sr1 * sr2 * min(sl1, sl2))
    # if sl1 * sl2 * sr1 * sr2 < sinter
    #     e11, e12, e21, e22 = E_bond.e11, E_bond.e12, E_bond.e21, E_bond.e22
    #     @cast E[(l1, l2), (r1, r2)] := e11[l1, r1] + e21[l2, r1] + e12[l1, r2] + e22[l2, r2]
    #     return  exp.(-beta * E) * message
    # elseif sr1 <= sr2 && sl1 <= sl2
    #     message = message'
    #     message = message .* (exp.(-beta * e21))'
    #     message = message * (exp.(-beta * e22))
    #     message .*= (exp.(-beta * e11))
    #     message .*= (exp.(-beta * e12))
    # elseif sr1 <= sr2 && sl2 <= sl1
    #     message = message'
    #     message = message .* (exp.(-beta * e11))'
    #     message = message * (exp.(-beta * e12))
    #     message .*= (exp.(-beta * e21))
    #     message .*= (exp.(-beta * e22))
    # elseif sr2 <= sr1 && sl1 <= sl2
    #     message = message'
    #     message = message .* (exp.(-beta * e22))'
    #     message = message * (exp.(-beta * e21))
    #     message .*= (exp.(-beta * e11))
    #     message .*= (exp.(-beta * e12))
    # else # sr2 <= sr1 && sl2 <= sl1
    #     message = message'
    #     message = message .* (exp.(-beta * e12))'
    #     message = message * (exp.(-beta * e11))
    #     message .*= (exp.(-beta * e21))
    #     message .*= (exp.(-beta * e22))
    # end
    # reshape(message, sr1 * sr2)
end


function factor_graph_2site(fg::LabelledGraph{S, T}, beta::Real) where {S, T}

    unified_vertices = unique([vertex[1:2] for vertex in vertices(fg)])
    new_fg = LabelledGraph{MetaDiGraph}(unified_vertices)

    vertx = Set()
    for v in vertices(fg)
        i, j, _ = v
        if (i, j) ∈ vertx continue end
        E1 = local_energy(fg, (i, j, 1))
        E2 = local_energy(fg, (i, j, 2))
        E = energy_2site(fg, i, j) .+ reshape(E1, :, 1) .+ reshape(E2, 1, :)
        sp = Spectrum(reshape(E, :), [])
        set_props!(new_fg, (i, j), Dict(:spectrum => sp))
        push!(vertx, (i, j))
    end

    edge_states = Set()
    for e ∈ edges(fg)
        if e in edge_states continue end
        v, w = src(e), dst(e)
        v1, v2, _ = v
        w1, w2, _ = w

        if (v1, v2) == (w1, w2) continue end

        add_edge!(new_fg, (v1, v2), (w1, w2))

        E, pl, pr = merge_vertices(fg, beta, v, w)
        set_props!(new_fg, (v1, v2), (w1, w2), Dict(:pl => pl, :en => E, :pr => pr))
        push!(edge_states, sort([(v1, v2), (w1, w2)]))
    end
    new_fg
end

function merge_vertices(fg::LabelledGraph{S, T}, β::Real, node1::NTuple{3, Int64}, node2::NTuple{3, Int64}
    ) where {S, T}
    i1, j1, _ = node1
    i2, j2, _ = node2

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

    MergedEnergy(e11, e12, e21, e22), pl, pr
end

function local_energy(fg::LabelledGraph{S, T}, v::NTuple{3, Int64}) where {S, T}
    has_vertex(fg, v) ? get_prop(fg, v, :spectrum).energies : zeros(1)
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
