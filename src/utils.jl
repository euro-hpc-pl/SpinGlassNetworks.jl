export
    zephyr_to_linear

"""
Rewriten from Dwave-networkx
m - Grid parameter for the Zephyr lattice.
t - Tile parameter for the Zephyr lattice; must be even.
"""
function zephyr_to_linear(m::Int, t::Int, q::NTuple{5, Int})

    M = 2 * m + 1
    u, w, k, j, z = q
    ind = (((u * M + w) * t + k) * 2 + j) * m + z + 1

    ind
end