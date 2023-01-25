

length(lp::list_of_projectors, index::Int64)
    return length(lp.data[index])

Proj = Union{Vector{Int64}, Cuarray{Int64, 1}}

struct list_of_projectors
    data :: Dict{Int64, Vector{Int64}}   #  tu kazdy klucz ma element
    dataGPU :: Dict{Int64, CuArray{Int64, 1}}
    ## jak kontrolowac czy okiekt na gpu czy na cpu?


get_projector(lp :: list_of_projectors, index::Int64, onGPU::bool)
return lp.data[index]  #  dodac element do dataGPU jak jest na cpu, ale nie na gpu ???

add_projector(lp :: list_of_projectors, p::Proj)
if p in values(lp.data)
    key = key of p_in_lp.data
else
    key = generate_new_key
    add!(lp.data, key => p)
return key
