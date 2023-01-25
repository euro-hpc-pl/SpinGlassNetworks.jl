

Proj = Union{Vector{Int64}, Cuarray{Int64, 1}}


struct list_of_projectors
    default_decive = CPU
    data :: Dict{devices, Dict{Int64, Proj}}   #  kazdy klucz ma element na cpu
    ## jak kontrolowac czy okiekt na gpu czy na cpu?
    # osobne miejsce 

length(lp::list_of_projectors, index::Int64) = length(lp.data[lp.default_device][index])


function clear_memory(lp, device)

function get_projector(lp :: list_of_projectors, index::Proj_keys, device::Int64; view::Symbol = :1D)
    # sprawdz czy index in lp.data[device]; jesli nie to go dodaj biorac dane z lp.data[default_device]
    return lp.data[device][index]  #  dodac element do dataGPU jak jest na cpu, ale nie na gpu ???
end


function get_projector_CSC(lp :: list_of_projectors, index::Proj_keys, device::Int64)
    # sprawdz czy index in lp.data[device]; jesli nie to go dodaj biorac dane z lp.data[default_device]
    return lp.data[device][index]  #  dodac element do dataGPU jak jest na cpu, ale nie na gpu ???
end


function add_projector(lp :: list_of_projectors, p::Proj)
if p in values(lp.data)
    key = key of p_in_lp.data
else
    key = generate_new_key
    add!(lp.data, key => p)
return key
end