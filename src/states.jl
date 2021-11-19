export idx, local_basis

idx(σ::Int) = (σ == -1) ? 1 : σ + 1
local_basis(d::Int) = union(-1, 1:d-1)

function all_states(rank::Union{Vector, NTuple})
    basis = [local_basis(r) for r ∈ rank]
    Iterators.product(basis...)
end
