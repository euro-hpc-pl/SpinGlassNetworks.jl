using LabelledGraphs

export 
    linear_to_pegasus,
    pegasus_to_nice,
    linear_to_nice,
    nice_to_dattani,
    dattani_to_linear,
    droplets_to_pegasus_nice


const Instance = Union{String, Dict}


"""
Rewriten from Dwave-networkx
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

    t = (2-u-(2*u-1)*floor(k/4)) % 3
    #Python has this weir bechavior that x%y is always nonnegative
    if t < 0
        t = t+3
    end

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
function linear_to_nice(size::Int, r::Int)
    pegasus_to_nice(linear_to_pegasus(size, r))
end


function droplets_to_pegasus_nice(instance::Instance)
    ig = ising_graph(instance)
    
    h_tuple = Dict()

end

"""
Changes pegasus_nice tuple into tuple from Dattani's paper. 
It is worth noting that two last indexes in Dattani's paper corespond to binary representatation of number from [0,1,2,3]
"""
function nice_to_dattani(pn::NTuple{5, Int})
    (pn[3], pn[2], pn[1], pn[4], pn[5])
end

"""
Changes tuple into linear index inspired by Dattani's paper
"""
function dattani_to_linear(size::Int, d::NTuple{5, Int})
    24*(size - 1) * d[1] + 24 * d[2] + 8 * d[3] + 4 * d[4] + d[5] + 1
end