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
    map = Dict(old[k, j, i] => (i, j, 1)  for i=1:m, j=1:n, k ∈ [1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19, 20])
    for i=1:m, j=1:n, k ∈ [5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24]
        push!(map, old[k, j, i] => (i, j, 2))
    end
    map
end