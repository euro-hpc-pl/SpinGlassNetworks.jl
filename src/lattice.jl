export
    super_square_lattice,
    pegasus_lattice,
    pegasus_lattice_masoud,
    pegasus_lattice_tomek,
    j_function,
    zephyr_lattice,
    zephyr_lattice_5tuple,
    zephyr_lattice_5tuple_rotated,
    periodic_lattice

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
    map = Dict(old[k, j, i] => (i, j, 1) for i=1:m, j=1:n, k ∈ (p * 8 + q for p ∈ 0 : t-1, q ∈ 1:4))
    for i=1:m, j=1:n, k ∈ (p * 8 + q for p ∈ 0 : t-1, q ∈ 5:8)
        push!(map, old[k, j, i] => (i, j, 2))
    end
    map
end

# TODO masoud / tomek should be removed from function names
function pegasus_lattice_masoud(size::NTuple{3, Int})
    m, n, t = size  # t is number of chimera units
    old = LinearIndices((1:8*t, 1:n, 1:m))
    map = Dict(old[k, j, i] => (i, j, 2) for i=1:m, j=1:n, k ∈ (p * 8 + q for p ∈ 0 : t-1, q ∈ 1:4))
    for i=1:m, j=1:n, k ∈ (p * 8 + q for p ∈ 0 : t-1, q ∈ 5:8)
        push!(map, old[k, j, i] => (i, j, 1))
    end
    map
end

function pegasus_lattice_tomek(size::NTuple{3, Int})
    m, n, t = size  # t is number of chimera units
    old = LinearIndices((1:8*t, 1:n, 1:m))
    map = Dict(old[k, j, i] => (i, n-j+1, 2) for i=1:m, j=1:n, k ∈ (p * 8 + q for p ∈ 0 : t-1, q ∈ 1:4))
    for i=1:m, j=1:n, k ∈ (p * 8 + q for p ∈ 0 : t-1, q ∈ 5:8)
        push!(map, old[k, j, i] => (i, n-j+1, 1))
    end
    map
end

function zephyr_lattice_z1(size::NTuple{3, Int})
    m, n, t = size # t is identical to dwave (Tile parameter for the Zephyr lattice)
    map = Dict{Int, NTuple{3, Int}}()

    for i=1:2*n, j ∈ 1:2*m
        for p in p_func(i, j, t, n, m)
            push!(map, (i-1)*(2*n*t) + (j-1)*(2*m*t) + p*n + (i-1)*(j%2) + 1  => (i, j, 1))
        end

        for q ∈ q_func(i, j, t, n, m)
            push!(map, 2*t*(2*n+1) + (i-1)*(2*n*t) + (j%2)*(2*m*t) + q*m + (j-1)*(i-1) + 1 => (i, j, 2))
        end
    end
    map
end

function j_function(i::Int, n::Int)
    i ∈ collect(1:n) && return collect((n + 1 - i):(n + i))
    collect((i-n):(3*n + 1 - i))
end

zephyr_lattice(size::NTuple{2, Int}) = zephyr_lattice((size[1], size[2], 4))

function zephyr_lattice(size::NTuple{3, Int})
    m, n, t = size
    zephyr_lattice_5tuple_rotated(m+1, n+1, zephyr_lattice_5tuple((Int(m/2), Int(n/2), t)))
end

function zephyr_lattice_5tuple(size::NTuple{3, Int})
    m, n , t = size # t is identical to dwave (Tile parameter for the Zephyr lattice)
    map = Dict{Int, NTuple{3, Int}}()

    # ( u, w, k, ζ, z)

    for u = 0
        for w ∈ 0:2:2*m, k ∈ 0:t-1, ζ ∈ 0:1, (i,z) ∈ enumerate(0:n-1)
            push!(map, zephyr_to_linear(m, t, (u,w,k,ζ,z)) => (2*i, w + 1, 1))
        end
        for w ∈ 1:2:2*m, k ∈ 0:t-1, ζ ∈ 0:1, z ∈ 0:n-1
            push!(map, zephyr_to_linear(m, t, (u,w,k,ζ,z)) => (2*z + 2*ζ + 1, w + 1, 1))
        end
    end

    for u = 1
        for w ∈ 0:2:2*m, k ∈ 0:t-1, ζ ∈ 0:1, (i,z) ∈ enumerate(0:n-1)
            push!(map, zephyr_to_linear(m, t, (u,w,k,ζ,z)) => (w + 1, 2*i, 2))
        end
        for w ∈ 1:2:2*m, k ∈ 0:t-1, ζ ∈ 0:1, z ∈ 0:n-1
            push!(map, zephyr_to_linear(m, t, (u,w,k,ζ,z)) => (w + 1, 2*z + 2*ζ + u, 2))
        end
    end
    map

end

function rotate(m::Int, n::Int)
    new_dict = Dict{NTuple{3, Int}, NTuple{3, Int}}()
    for (k,j) ∈ enumerate(1:2:m)
        for (l,i) ∈ enumerate(n-1:-2:1)
            push!(new_dict, (i,j,1) => (i/2 + k - 1, l + k - 1, 1))
            push!(new_dict, (i,j,2) => (i/2 + k - 1, l + k - 1, 2))
        end
    end

    for (k,j) ∈ enumerate(2:2:m)
        for (l,i) ∈ enumerate(n:-2:1)
            push!(new_dict, (i,j,1) => ((i-1)/2 + k, l + k - 1, 1))
            push!(new_dict, (i,j,2) => ((i-1)/2 + k, l + k - 1, 2))
        end
    end
    new_dict
end

function empty_clusters(m::Int, n::Int)
    p = (m - 1) / 2
    count = 0
    ii = []
    for (i, j) ∈ enumerate(1:p-1)
        count += i
        push!(ii, i)
    end
    (count, reverse(ii))
end

function zephyr_lattice_5tuple_rotated(m::Int, n::Int, map::Dict{Int, NTuple{3, Int}})
    rotated_map = rotate(m, n) #5, 5, for Z2
    new_map = Dict{Int, NTuple{3, Int}}()

    (empty, ii) = empty_clusters(m, n)
    for k in keys(map)
        push!(new_map, k => rotated_map[map[k]])
    end

    empty_vertices = empty_indexing(m,n)
    for (k,l) ∈ enumerate(empty_vertices)
        push!(new_map, -k => l)
    end
    new_map
end

function empty_indexing(m::Int, n::Int)
    (empty, ii) = empty_clusters(m, n)
    p = Int((m-1)/2)
    empty_vertices = []
    for (k,l) ∈ enumerate(ii)
        for i ∈ 1:l
            push!(empty_vertices, (k, i, 1))
            push!(empty_vertices, (k, i, 2))
            push!(empty_vertices, (k, i + m - p + k - 1, 1))
            push!(empty_vertices, (k, i + m - p + k - 1, 2))
        end
    end
    for (k,l) ∈ enumerate(reverse(ii))
        for i ∈ 1:l
            push!(empty_vertices, (k + m - p, i, 1))
            push!(empty_vertices, (k + m - p, i, 2))
            push!(empty_vertices, (k + m - p, ii[k] + m - p + k - i, 1))
            push!(empty_vertices, (k + m - p, ii[k] + m - p + k - i, 2))
        end
    end
    empty_vertices
end

function periodic_lattice(size::NTuple{3, Int})
    m, n, t = size
    map = super_square_lattice((m, n, t))
    new_map  = Dict{Int, NTuple{2, Int}}()
    for (key, val) ∈ map
        i, j = val
        if i <= m/2
            if j <= m/2
                push!(new_map, key => (i, j))
            elseif j > m/2
                push!(new_map, key => (i, m-j+1))
            end
        elseif i > m/2
            if j <= m/2
                push!(new_map, key => (m-i+1, j))
            elseif j > m/2
                push!(new_map, key => (m-i+1, m-j+1))
            end
        end
    end
    new_map
end
