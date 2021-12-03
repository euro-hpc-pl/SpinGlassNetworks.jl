export idx, local_basis

idx(σ::Int) = (σ == -1) ? 1 : σ + 1
local_basis(d::Int) = union(-1, 1:d-1)

all_states(rank::Union{Vector, NTuple}) = Iterators.product(local_basis.(rank)...)
