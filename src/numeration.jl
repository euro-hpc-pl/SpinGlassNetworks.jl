using LabelledGraphs

export 
    linear_to_pegasus,
    pegasus_to_nice,
    linear_to_nice,
    droplets_to_pegasus_nice


const Instance = Union{String, Dict}
const IsingGraph = LabelledGraph{MetaGraph{Int64, Float64}, Int64}

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
function pegasus_to_nice(p::NTuple{4, Int64}) #TODO: Write it better. Now it is rewriten verbatim.
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

function linear_to_nice(size::Int, r::Int)
    pegasus_to_nice(linear_to_pegasus(size, r))
end

function droplets_to_pegasus_nice(instance::Instance)
    ig = ising_graph(instance)
    
    h_tuple = Dict()

end

