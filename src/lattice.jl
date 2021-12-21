export super_square_lattice, pegasus_lattice

 function super_square_lattice(size::NTuple{5, Int})
    m, um, n, un, t = size
    old = LinearIndices((1:t, 1:un, 1:n, 1:um, 1:m))
    Dict(old[k, uj, j, ui, i] => (i, j) for i=1:m, ui=1:um, j=1:n, uj=1:un, k=1:t)
end

function super_square_lattice(size::NTuple{3, Int})
    m, n, t = size
    super_square_lattice((m, 1, n, 1, t))
end

function pegasus_lattice(size::NTuple{2, Int})
    m, n = size
    old = LinearIndices((1:24, 1:n, 1:m))
    map = Dict(old[k, j, i] => (i, j, 1) for i=1:m, j=1:n, k ∈ vcat(1:4, 9:12, 17:20))
    for i=1:m, j=1:n, k ∈ vcat(5:8, 13:16, 21:24) push!(map, old[k, j, i] => (i, j, 2)) end
    map
end

function pegasus_lattice(size::NTuple{3, Int})
    m, n, t = size  # t is number of chimera units
    old = LinearIndices((1:8*t, 1:n, 1:m))
    map = Dict(
        old[k, j, i] => (i, j, 1) for i=1:m, j=1:n, k ∈ (n * 8 + m for n ∈ 0:t-1, m ∈ 1:4)
    )
    for i=1:m, j=1:n, k ∈ (n * 8 + m for n ∈ 0:t-1, m ∈ 5:8)
        push!(map, old[k, j, i] => (i, j, 2))
    end
    map
end
