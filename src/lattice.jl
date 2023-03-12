export super_square_lattice

function super_square_lattice(size::NTuple{5,Int})
    m, um, n, un, t = size
    old = LinearIndices((1:t, 1:un, 1:n, 1:um, 1:m))

    Dict(old[k, uj, j, ui, i] => (i, j) for i = 1:m, ui = 1:um, j = 1:n, uj = 1:un, k = 1:t)
end

"""
    super_square_lattice(size::NTuple{3,Int})

Create cluster assignment rule for Chimera architecture. The input is asumed to have form `(m, n, t)` where `m` is number of
columns, `n` number of rows and `t` is the size of the shore within each Chimera tile 
([details](https://docs.ocean.dwavesys.com/en/stable/docs_dnx/reference/generated/dwave_networkx.chimera_graph.html)). 
If `t=1` then this assigment rule can be used for square lattice, where every site will form its own unit cell.  
"""
function super_square_lattice(size::NTuple{3,Int})
    m, n, t = size
    super_square_lattice((m, 1, n, 1, t))
end
