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


function zephyr_lattice(size::NTuple{3, Int})::Dict{Int, NTuple{3, Int}}
m, n , t = size # t is identical to dwave (Tile parameter for the Zephyr lattice)
map = Dict()

for i=1:2*n, j=1:2*m
    for p in p_func(i, j, t)
        push!(map, (i-1)*(2*n*t) + (j-1)*(2*m*t) + p*n + (i-1)*(j%2) => (i,j,1))
    end
    
    for q in q_func(i, j, t)
        push!(map, 2*t*(2*n+1) + (i-1)*(2*n*t) + (j%2)*(2*m*t) + q*m + (j-1)*(i-1) => (i,j,2))
    end
   
end
map
end

function flag_horizontal(i::Int, j::Int)
    (i+j)%2 == 1 ? true : false 
end

function flag_vertical(i::Int, j::Int)
    (i+j)%2 == 0 ? true : false 
end


function p_func(i::Int, j::Int, t::Int)
    flag_horizontal(i, j) ? collect(1:2:2*t) : collect(1:2*t)
end

function q_func(i::Int, j::Int, t::Int)
    flag_vertical(i, j) ? collect(1:2:2*t) : collect(1:2*t)
end
