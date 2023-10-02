
export
    belief_propagation,
    clustered_hamiltonian_2site,
    projector,
    get_neighbors,
    MergedEnergy,
    update_message,
    merge_vertices_cl_h,
    local_energy,
    interaction_energy,
    SparseCSC 

"""
Perform belief propagation on a given clustered hamiltonian.

# Arguments:
- `cl_h::LabelledGraph{S, T}`: The clustered hamiltonian represented as a labeled graph.
- `beta::Real`: The temperature parameter for the belief propagation algorithm.
- `tol::Real (optional, default=1e-6)`: The convergence tolerance. The algorithm stops when the message updates between iterations are smaller than this value.
- `iter::Int (optional, default=1)`: The maximum number of iterations to perform.

# Returns:
- `beliefs::Dict`: A dictionary where keys are vertices of clustered hamiltonian, and values are the resulting beliefs after belief propagation.

The function implements belief propagation on the given clustered hamiltonian `cl_h` to calculate beliefs for each vertex.
Belief propagation is an iterative algorithm that computes beliefs by passing messages between vertices and edges of the clustered hamiltonian. 
The algorithm continues until convergence or until the specified maximum number of iterations is reached.
The beliefs are computed based on the temperature parameter `beta`, which controls the influence of energy values on the beliefs.
"""
function belief_propagation(cl_h::LabelledGraph{S, T}, beta::Real; tol=1e-6, iter=1)  where {S, T}
    messages_ve = Dict()
    messages_ev = Dict()

    # Initialize messages with uniform probabilities
    for v in vertices(cl_h)
        for (n, pv, _) in get_neighbors(cl_h, v)
            push!(messages_ev, (n, v) => ones(maximum(pv)))
        end
    end

    # Perform message passing until convergence
    converged = false
    iteration = 0
    while !converged && iteration < iter  # Set an appropriate number of iterations and convergence threshold
        iteration += 1
        old_messages_ev = deepcopy(messages_ev)
        for v in vertices(cl_h)
            #update messages from vertex to edge
            node_messages = Dict()
            for (n1, pv1, _) ∈ get_neighbors(cl_h, v)
                node_messages[n1, v] = messages_ev[n1, v][pv1]
            end
            for (n1, pv1, _) ∈ get_neighbors(cl_h, v)
                E_local = get_prop(cl_h, v, :spectrum).energies
                temp = exp.(-(E_local .- minimum(E_local)) * beta)
                for (n2, pv2, _) in get_neighbors(cl_h, v)
                    if n1 == n2 continue end
                    temp .*= node_messages[n2, v] # messages_ev[n2, v][pv2]
                end
                temp ./= sum(temp)
                messages_ve[v, n1] = SparseCSC(eltype(temp), pv1) * temp
            end
        end

        #update messages from edge to vertex
        for v in vertices(cl_h)
            for (n, _, en) ∈ get_neighbors(cl_h, v)
                messages_ev[n, v] = update_message(en, messages_ve[n, v], beta)
            end
        end

        # Check convergence
        converged = all([all(abs.(old_messages_ev[v] .- messages_ev[v]) .< tol) for v in keys(messages_ev)])
    end

    beliefs = Dict()
    for v in vertices(cl_h)
        E_local = get_prop(cl_h, v, :spectrum).energies
        beliefs[v] = exp.(-E_local * beta)
        for (n, pv, _) ∈ get_neighbors(cl_h, v)
            beliefs[v] .*= messages_ev[n, v][pv]
        end
        beliefs[v] = -log.(beliefs[v])./beta
        beliefs[v] = beliefs[v] .- minimum(beliefs[v])
    end

    beliefs
end

"""
Returns the neighbors of a given vertex in a clustered Hamiltonian.

# Arguments:
- `cl_h::LabelledGraph{S, T}`: The clustered Hamiltonian represented as a labeled graph.
- `vertex::NTuple`: The vertex for which neighbors are to be retrieved.

# Returns:
- `neighbors::Vector{Tuple}`: A vector of tuples representing the neighbors of the specified vertex. Each tuple contains the following information:
    - `dst_node::T`: The neighboring vertex.
    - `pv::Matrix`: The projector associated with the edge connecting the vertex and its neighbor.
    - `en::Real`: The energy associated with the edge connecting the vertex and its neighbor.

This function retrieves the neighbors of a given vertex in a clustered Hamiltonian graph. It iterates through the edges of the graph and identifies edges connected to the specified vertex. For each neighboring edge, it extracts and returns the neighboring vertex, the associated projector, and the energy.
"""
function get_neighbors(cl_h::LabelledGraph{S, T}, vertex::NTuple) where {S, T}
    neighbors = []
    for edge in edges(cl_h)
        src_node, dst_node = src(edge), dst(edge)
        if src_node == vertex
            en = get_prop(cl_h, src_node, dst_node, :en)
            idx_pv = get_prop(cl_h, src_node, dst_node, :ipl)
            pv = get_projector!(get_prop(cl_h, :pool_of_projectors), idx_pv, :CPU)
            push!(neighbors, (dst_node, pv, en))
        elseif dst_node == vertex
            en = get_prop(cl_h, src_node, dst_node, :en)'
            idx_pv = get_prop(cl_h, src_node, dst_node, :ipr)
            pv = get_projector!(get_prop(cl_h, :pool_of_projectors), idx_pv, :CPU)
            push!(neighbors, (src_node, pv, en))
        end
    end
    return neighbors
end

"""
A custom Julia struct representing energy values in a merged format for use in specific calculations.

# Fields:
- `e11::AbstractMatrix{T}`
- `e12::AbstractMatrix{T}`
- `e21::AbstractMatrix{T}`
- `e22::AbstractMatrix{T}`
    
The `MergedEnergy` struct is used to represent energy values that are organized in a merged format. This format is often utilized in certain computational tasks, where energy values are categorized based on combinations of left and right factors.
    
Each field of the `MergedEnergy` struct stores energy values as an `AbstractMatrix{T}` of type `T`, where `T` is a subtype of the `Real` abstract type. The specific organization and interpretation of these energy values depend on the context in which this struct is used.    
"""
struct MergedEnergy{T <: Real}
    e11::AbstractMatrix{T}
    e12::AbstractMatrix{T}
    e21::AbstractMatrix{T}
    e22::AbstractMatrix{T}
end

Base.adjoint(s::MergedEnergy) = MergedEnergy(s.e11', s.e21', s.e12', s.e22')

"""
Update a message using energy values and temperature.

# Arguments:
- `E_bond::AbstractArray`: An array of energy values associated with a bond or interaction.
- `message::Vector`: The input message vector to be updated.
- `beta::Real`: The temperature parameter controlling the influence of energy values.

# Returns:
- `updated_message::Vector`: The updated message vector after applying the energy-based update.

This function takes energy values `E_bond` associated with a bond or interaction, an input message vector `message`, and a temperature parameter `beta`. It updates the message by first adjusting the energy values relative to their minimum value, exponentiating them with a negative sign and scaling by `beta`, and then multiplying them element-wise with the input message.

The result is an updated message that reflects the influence of energy values and temperature.
"""
function update_message(E_bond::AbstractArray, message::Vector, beta::Real)
    E_bond = E_bond .- minimum(E_bond)
    exp.(-beta * E_bond) * message
end

"""
Update a message using energy values and temperature in a merged energy format.

# Arguments:
- `E_bond::MergedEnergy`: An instance of the `MergedEnergy` type representing energy values for the bond or interaction.
- `message::Vector`: The input message vector to be updated.
- `beta::Real`: The temperature parameter controlling the influence of energy values.

# Returns:
- `updated_message::Vector`: The updated message vector after applying the energy-based update.

This function takes energy values `E_bond` in a merged energy format, an input message vector `message`, and a temperature parameter `beta`. It updates the message based on the energy values and temperature using a specified algorithm.

The `MergedEnergy` type represents energy values in a merged format, and the function processes these values accordingly to update the message vector.
"""
function update_message(E_bond::MergedEnergy, message::Vector, beta::Real)
    e11, e12, e21, e22 = E_bond.e11, E_bond.e12, E_bond.e21, E_bond.e22
    # equivalent to
    # @cast E[(l1, l2), (r1, r2)] := e11[l1, r1] + e21[l2, r1] + e12[l1, r2] + e22[l2, r2]
    # exp.(-beta * E) * message

    e11 = exp.(-beta .* (e11 .- minimum(e11)))
    e12 = exp.(-beta .* (e12 .- minimum(e12)))
    e21 = exp.(-beta .* (e21 .- minimum(e21)))
    e22 = exp.(-beta .* (e22 .- minimum(e22)))
    sl1, sl2, sr1, sr2 = size(e11, 1), size(e21, 1), size(e21, 2), size(e22, 2)

    if sl1 * sl2 * sr1 * sr2 < max(sr1 * sr2 * min(sl1, sl2), sl1 * sl2 * min(sr1, sr2))
        R = reshape(e11, sl1, 1, sr1, 1) .* reshape(e21, 1, sl2, sr1, 1)
        R = R .* reshape(e12, sl1, 1, 1, sr2)
        R = R .* reshape(e22, 1, sl2, 1, sr2)
        R = reshape(R, sl1 * sl2, sr1 * sr2) * message
    elseif sl1 <= sl2 && sr1 <= sr2
        R = reshape(e12, sl1, 1, sr2) .* reshape(message, 1, sr1, sr2)
        R = reshape(reshape(R, sl1 * sr1, sr2) * e22', sl1, sr1, sl2)  # [l1, r1, l2]
        R .*= reshape(e11, sl1, sr1, 1)  # [l1, r1, l2] .* [l1, r1, :]
        R .*= reshape(e21', 1, sr1, sl2)  # [l1, r1, l2] .* [:, r1, l2]
        R = reshape(sum(R, dims=2), sl1 * sl2)
    elseif sl1 <= sl2 && sr2 <= sr1
        R = reshape(e11', sr1, sl1, 1) .* reshape(message, sr1, 1, sr2)
        R = reshape(e21 * reshape(R, sr1, sl1 * sr2), sl2, sl1, sr2)
        R .*= reshape(e12, 1, sl1, sr2)  # [l2, l1, r2] .* [:, l1, r2]
        R .*= reshape(e22, sl2, 1, sr2)  # [l2, l1, r2] .* [l2, :, r2]
        R = reshape(reshape(sum(R, dims=3), sl2, sl1)', sl1 * sl2)
    elseif sl2 <= sl1 && sr1 <= sr2
        R = reshape(e22, sl2, 1, sr2) .* reshape(message, 1, sr1, sr2)
        R = reshape(reshape(R, sl2 * sr1, sr2) * e12', sl2, sr1, sl1)  # [l2, r1, l1]
        R .*= reshape(e11', 1, sr1, sl1)  # [l2, r1, l1] .* [:, r1, l1]
        R .*= reshape(e21, sl2, sr1, 1)   # [l2, r1, l1] .* [l2, r1, :]
        R = reshape(reshape(sum(R, dims=2), sl2, sl1)', sl1 * sl2)
    else # sl2 <= sl1 && sr2 <= sr1
        R = reshape(e21', sr1, sl2, 1) .* reshape(message, sr1, 1, sr2)
        R = reshape(e11 * reshape(R, sr1, sl2 * sr2), sl1, sl2, sr2)
        R .*= reshape(e12, sl1, 1, sr2)  # [l1, l2, r2] .* [l1, :, r2]
        R .*= reshape(e22, 1, sl2, sr2)  # [l1, l2, r2] .* [:, l2, r2]
        R = reshape(sum(R, dims=3), sl1 * sl2)
    end
    R
end

"""
Constructs a clustered Hamiltonian for a given clustered Hamiltonian with a 2-site cluster approximation.

# Arguments:
- `cl_h::LabelledGraph{S, T}`: The clustered Hamiltonian represented as a labeled graph.
- `beta::Real`: The temperature parameter for the 2-site cluster Hamiltonian construction.
    
# Returns:
- `new_cl_h::LabelledGraph{MetaDiGraph}`: A new labeled graph representing the 2-site cluster Hamiltonian.
    
 This function constructs a clustered Hamiltonian `cl_h` by applying a 2-site cluster approximation. It combines and merges vertices and edges of the original graph to create a simplified representation of the Hamiltonian.
    
The resulting `new_cl_h` graph represents the 2-site cluster Hamiltonian with simplified interactions between clusters. The energy values, projectors, and spectra associated with the new vertices and edges are computed based on the provided temperature parameter `beta`.
"""
function clustered_hamiltonian_2site(cl_h::LabelledGraph{S, T}, beta::Real) where {S, T}

    unified_vertices = unique([vertex[1:2] for vertex in vertices(cl_h)])
    new_cl_h = LabelledGraph{MetaDiGraph}(unified_vertices)
    new_lp = PoolOfProjectors{Int}()

    vertx = Set()
    for v in vertices(cl_h)
        i, j, _ = v
        if (i, j) ∈ vertx continue end
        E1 = local_energy(cl_h, (i, j, 1))
        E2 = local_energy(cl_h, (i, j, 2))
        E = energy_2site(cl_h, i, j) .+ reshape(E1, :, 1) .+ reshape(E2, 1, :)
        sp = Spectrum(reshape(E, :), [], [])
        set_props!(new_cl_h, (i, j), Dict(:spectrum => sp))
        push!(vertx, (i, j))
    end

    edge_states = Set()
    for e ∈ edges(cl_h)
        if e in edge_states continue end
        v, w = src(e), dst(e)
        v1, v2, _ = v
        w1, w2, _ = w

        if (v1, v2) == (w1, w2) continue end

        add_edge!(new_cl_h, (v1, v2), (w1, w2))

        E, pl, pr = merge_vertices_cl_h(cl_h, beta, v, w)
        ipl = add_projector!(new_lp, pl)
        ipr = add_projector!(new_lp, pr)
        set_props!(new_cl_h, (v1, v2), (w1, w2), Dict(:ipl => ipl, :en => E, :ipr => ipr))
        push!(edge_states, sort([(v1, v2), (w1, w2)]))
    end
    set_props!(new_cl_h, Dict(:pool_of_projectors => new_lp))
    new_cl_h
end

"""
Merge two vertices in a clustered Hamiltonian to create a single merged vertex.

# Arguments:
- `cl_h::LabelledGraph{S, T}`: The clustered Hamiltonian represented as a labeled graph.
- `β::Real`: The temperature parameter controlling the influence of energy values.
- `node1::NTuple{3, Int64}`: The coordinates of the first vertex to merge.
- `node2::NTuple{3, Int64}`: The coordinates of the second vertex to merge.

# Returns:
- `merged_energy::MergedEnergy`: An instance of the `MergedEnergy` type representing the merged energy values.
- `pl::AbstractVector`: The merged left projector.
- `pr::AbstractVector`: The merged right projector.

This function merges two vertices in a clustered Hamiltonian graph `cl_h` to create a single merged vertex. The merging process combines projectors and energy values associated with the original vertices based on the provided temperature parameter `β`.

The merged energy values, left projector `pl`, and right projector `pr` are computed based on the interactions between the original vertices and their respective projectors.
"""
function merge_vertices_cl_h(cl_h::LabelledGraph{S, T}, β::Real, node1::NTuple{3, Int64}, node2::NTuple{3, Int64}
    ) where {S, T}
    i1, j1, _ = node1
    i2, j2, _ = node2

    p21l = projector(cl_h, (i1, j1, 2), (i2, j2, 1))
    p22l = projector(cl_h, (i1, j1, 2), (i2, j2, 2))
    p12l = projector(cl_h, (i1, j1, 1), (i2, j2, 2))
    p11l = projector(cl_h, (i1, j1, 1), (i2, j2, 1))

    p1l, (p11l, p12l) = fuse_projectors((p11l, p12l))
    p2l, (p21l, p22l) = fuse_projectors((p21l, p22l))

    p11r = projector(cl_h, (i2, j2, 1), (i1, j1, 1))
    p21r = projector(cl_h, (i2, j2, 1), (i1, j1, 2))
    p12r = projector(cl_h, (i2, j2, 2), (i1, j1, 1))
    p22r = projector(cl_h, (i2, j2, 2), (i1, j1, 2))

    p1r, (p11r, p21r) = fuse_projectors((p11r, p21r))
    p2r, (p12r, p22r) = fuse_projectors((p12r, p22r))

    pl = outer_projector(p1l, p2l)
    pr = outer_projector(p1r, p2r)

    e11 = interaction_energy(cl_h, (i1, j1, 1), (i2, j2, 1))
    e12 = interaction_energy(cl_h, (i1, j1, 1), (i2, j2, 2))
    e21 = interaction_energy(cl_h, (i1, j1, 2), (i2, j2, 1))
    e22 = interaction_energy(cl_h, (i1, j1, 2), (i2, j2, 2))

    e11 = e11[p11l, p11r]
    e21 = e21[p21l, p21r]
    e12 = e12[p12l, p12r]
    e22 = e22[p22l, p22r]

    MergedEnergy(e11, e12, e21, e22), pl, pr
end

"""
Get the local energy associated with a vertex in a clustered Hamiltonian.

# Arguments:
- `cl_h::LabelledGraph{S, T}`: The clustered Hamiltonian represented as a labeled graph.
- `v::NTuple{3, Int64}`: The coordinates of the vertex for which the local energy is requested.

# Returns:
- `local_energy::AbstractVector`: An abstract vector containing the local energy values associated with the specified vertex.

This function retrieves the local energy values associated with a given vertex `v` in a clustered Hamiltonian graph `cl_h`. If the vertex exists in the graph and has associated energy values, it returns those values; otherwise, it returns a vector of zeros.

The local energy values are typically obtained from the spectrum associated with the vertex.
"""
function local_energy(cl_h::LabelledGraph{S, T}, v::NTuple{3, Int64}) where {S, T}
    has_vertex(cl_h, v) ? get_prop(cl_h, v, :spectrum).energies : zeros(1)
end

"""
Get the interaction energy between two vertices in a clustered Hamiltonian.

# Arguments:
- `cl_h::LabelledGraph{S, T}`: The clustered Hamiltonian represented as a labeled graph.
- `v::NTuple{3, Int64}`: The coordinates of the first vertex.
- `w::NTuple{3, Int64}`: The coordinates of the second vertex.

# Returns:
- `interaction_energy::AbstractMatrix`: An abstract matrix containing the interaction energy values between the specified vertices.

This function retrieves the interaction energy values between two vertices, `v` and `w`, in a clustered Hamiltonian graph `cl_h`. If there is a directed edge from `w` to `v`, it returns the corresponding energy values; if there is a directed edge from `v` to `w`, it returns the transpose of the energy values; otherwise, it returns a matrix of zeros.
The interaction energy values represent the energy associated with the interaction or connection between the two vertices.
"""
function interaction_energy(cl_h::LabelledGraph{S, T}, v::NTuple{3, Int64}, w::NTuple{3, Int64}) where {S, T}
    if has_edge(cl_h, w, v)
        get_prop(cl_h, w, v, :en)'
    elseif has_edge(cl_h, v, w)
        get_prop(cl_h, v, w, :en)
    else
        zeros(1, 1)
    end
end

"""
Get the projector associated with an edge between two vertices in a clustered Hamiltonian.

# Arguments:
- `cl_h::LabelledGraph{S, T}`: The clustered Hamiltonian represented as a labeled graph.
- `v::NTuple{N, Int64}`: The coordinates of one of the two vertices connected by the edge.
- `w::NTuple{N, Int64}`: The coordinates of the other vertex connected by the edge.

# Returns:
- `p::AbstractVector`: An abstract vector representing the projector associated with the specified edge.

This function retrieves the projector associated with an edge between two vertices, `v` and `w`, in a clustered Hamiltonian graph `cl_h`. If there is a directed edge from `w` to `v`, it returns the index of right projector (`:ipr`); if there is a directed edge from `v` to `w`, it returns the index of left projector (`:ipl`). If no edge exists between the vertices, it returns a vector of ones.
"""
function projector(cl_h::LabelledGraph{S, T}, v::NTuple{N, Int64}, w::NTuple{N, Int64}) where {S, T, N}
    if has_edge(cl_h, w, v)
        idx_p = get_prop(cl_h, w, v, :ipr)
        p = get_projector!(get_prop(cl_h, :pool_of_projectors), idx_p, :CPU)
    elseif has_edge(cl_h, v, w)
        idx_p = get_prop(cl_h, v, w, :ipl)
        p = get_projector!(get_prop(cl_h, :pool_of_projectors), idx_p, :CPU)
    else
        p = ones(Int, v ∈ vertices(cl_h) ? length(get_prop(cl_h, v, :spectrum).energies) : 1)
    end
end

function fuse_projectors(projectors::NTuple{N, K}) where {N, K}
    fused, transitions_matrix = rank_reveal(hcat(projectors...), :PE)
    transitions = Tuple(Array(t) for t ∈ eachcol(transitions_matrix))
    fused, transitions
end

function outer_projector(p1::Array{T, 1}, p2::Array{T, 1}) where T <: Number
    reshape(reshape(p1, :, 1) .+ maximum(p1) .* reshape(p2 .- 1, 1, :), :)
end

"""
Create a sparse column-compressed (CSC) matrix with specified column indices and values.

# Arguments:
- `::Type{R}`: The element type of the sparse matrix (e.g., `Float64`, `Int64`).
- `p::Vector{Int64}`: A vector of column indices for the non-zero values.

# Returns:
- `sparse_matrix::SparseMatrixCSC{R}`: A sparse column-compressed matrix with non-zero values at specified columns.

This constructor function creates a sparse column-compressed (CSC) matrix of element type `R` based on the provided column indices `p` and values. The resulting matrix has non-zero values at the specified column indices, while all other elements are zero.
The `SparseCSC` constructor is useful for creating sparse matrices with specific column indices and values efficiently.
"""
function SparseCSC(::Type{R}, p::Vector{Int64}) where R <: Real
    n = length(p)
    mp = maximum(p)
    cn = collect(1:n)
    co = ones(R, n)
    sparse(p, cn, co, mp, n)
end