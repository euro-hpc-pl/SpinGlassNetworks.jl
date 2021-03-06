export super_square_lattice

 function super_square_lattice(size::NTuple{5, Int})
    m, um, n, un, t = size
    old = LinearIndices((1:t, 1:un, 1:n, 1:um, 1:m))

    Dict(
            old[k, uj, j, ui, i] => (i, j)
            for i=1:m, ui=1:um, j=1:n, uj=1:un, k=1:t
    )
end

function super_square_lattice(size::NTuple{3, Int})
    m, n, t = size
    super_square_lattice((m, 1, n, 1, t))
end
