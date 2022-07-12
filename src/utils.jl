using LabelledGraphs

export
    linear_to_pegasus,
    pegasus_to_nice,
    linear_to_nice,
    nice_to_dattani,
    dattani_to_linear,
    droplets_to_pegasus_nice,
    linear_to_zephyr,
    zephyr_to_linear

    
const Instance = Union{String, Dict}

"""
Rewriten from Dwave-networkx lib
"""
function linear_to_pegasus(size::Int, r::Int)
    m, m1 = size, size - 1
    r, z = divrem(r, m1)
    r, k = divrem(r, 12)
    u, w = divrem(r, m)
    (u, w, k, z)
end

"""
Rewriten from Dwave-networkx
"""
function pegasus_to_nice(p::NTuple{4, Int}) #TODO: Write it better. Now it is rewriten verbatim.
    u, w, k, z = p

    t = (2 - u - (2 * u - 1) * floor(k / 4)) % 3
    if t < 0 t = t+3 end

    if t == 0
        if u > 0
            y = w-1
            x = z
        else
            y = z
            x = w
        end
        return (round(Int, t), y, x, u, k-4)

    elseif t == 1
        if u > 0
            y = w-1
            x = z
            k1 = k
        else
            y = z
            x = w
            k1 = k-8
        end
        return (round(Int, t), y, x, u, k1)

    elseif t == 2
        if u > 0
            y = w
            x = z
            k1 = k-8
        else
            y = z
            x = w-1
            k1 = k
        end
        return (round(Int, t), y, x, u, k1)
    end
end

"""
Rewriten from Dwave-networkx
"""
linear_to_nice(size::Int, r::Int) = pegasus_to_nice(linear_to_pegasus(size, r))

"""
Based on [1] https://arxiv.org/abs/1901.07636
Changes pegasus_nice tuple into tuple from [1].
It is worth noting that two last indexes in [1]
corespond to binary representatation of number from [0, 1, 2, 3]
"""
nice_to_dattani(pn::NTuple{5, Int}) = (pn[3], pn[2], pn[1], pn[4], pn[5])

"""
Changes tuple into linear index inspired by Dattani's paper
"""
function dattani_to_linear(size::Int, d::NTuple{5, Int})
    24 * (size - 1) * d[1] + 24 * d[2] + 8 * d[3] + 4 * d[4] + d[5] + 1
end

"""
Rewriten from Dwave-networkx
"""
function linear_to_zephyr(m::Int, t::Int, ind::Int)
    # m - Grid parameter for the Zephyr lattice.
    # t - Tile parameter for the Zephyr lattice; must be even.

    M = 2 * m + 1

    ind, z = divrem(ind, m)
    ind, j = divrem(ind, 2)
    ind, k = divrem(ind, t)
    u, w = divrem(ind, M)

    (u, w, k, j, z)
end

"""
Rewriten from Dwave-networkx
"""
function zephyr_to_linear(m::Int, t::Int, q::NTuple{5, Int})
    # m - Grid parameter for the Zephyr lattice.
    # t - Tile parameter for the Zephyr lattice; must be even.

    M = 2 * m + 1
    u, w, k, j, z = q
    ind = (((u * M + w) * t + k) * 2 + j) * m + z

    ind
end