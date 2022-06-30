export
    super_square_lattice,
    pegasus_lattice,
    pegasus_lattice_masoud,
    pegasus_lattice_tomek,
    zephyr_lattice
    
"Variable number of Ising graph -> Factor graph coordinate system"
function super_square_lattice(size::NTuple{5, Int})
    m, um, n, un, t = size
    old = LinearIndices((1:t, 1:un, 1:n, 1:um, 1:m))
    Dict(old[k, uj, j, ui, i] => (i, j) for i=1:m, ui=1:um, j=1:n, uj=1:un, k=1:t)
end

function super_square_lattice(size::NTuple{3, Int})
    m, n, t = size
    super_square_lattice((m, 1, n, 1, t))
end

pegasus_lattice(size::NTuple{2, Int}) = pegasus_lattice((size[1], size[2], 3))

function pegasus_lattice(size::NTuple{3, Int})
    m, n, t = size  # t is number of chimera units
    old = LinearIndices((1:8*t, 1:n, 1:m))
    map = Dict(
        old[k, j, i] => (i, j, 1) for i=1:m, j=1:n, k ∈ (p * 8 + q for p ∈ 0 : t-1, q ∈ 1:4)
    )
    for i=1:m, j=1:n, k ∈ (p * 8 + q for p ∈ 0 : t-1, q ∈ 5:8)
        push!(map, old[k, j, i] => (i, j, 2))
    end
    map
end

function pegasus_lattice_masoud(size::NTuple{3, Int})
    m, n, t = size  # t is number of chimera units
    old = LinearIndices((1:8*t, 1:n, 1:m))
    map = Dict(
        old[k, j, i] => (i, j, 2) for i=1:m, j=1:n, k ∈ (p * 8 + q for p ∈ 0 : t-1, q ∈ 1:4)
    )
    for i=1:m, j=1:n, k ∈ (p * 8 + q for p ∈ 0 : t-1, q ∈ 5:8)
        push!(map, old[k, j, i] => (i, j, 1))
    end
    map
end

function pegasus_lattice_tomek(size::NTuple{3, Int})
    m, n, t = size  # t is number of chimera units
    old = LinearIndices((1:8*t, 1:n, 1:m))
    map = Dict(
        old[k, j, i] => (i, n-j+1, 2) for i=1:m, j=1:n, k ∈ (p * 8 + q for p ∈ 0 : t-1, q ∈ 1:4)
    )
    for i=1:m, j=1:n, k ∈ (p * 8 + q for p ∈ 0 : t-1, q ∈ 5:8)
        push!(map, old[k, j, i] => (i, n-j+1, 1))
    end
    map
end