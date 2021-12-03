export all_states, local_basis, gibbs_tensor, brute_force
export full_spectrum, energy, Spectrum, idx, local_basis

idx(σ::Int) = (σ == -1) ? 1 : σ + 1
local_basis(d::Int) = union(-1, 1:d-1)
all_states(rank::Union{Vector, NTuple}) = Iterators.product(local_basis.(rank)...)

function gibbs_tensor(ig::IsingGraph, β::Real=1.0)
    states = collect.(all_states(rank_vec(ig)))
    ρ = exp.(-β .* energy.(states, Ref(ig)))
    ρ ./ sum(ρ)
end

energy(σ::Vector, J::Matrix, η::Vector=σ) = dot(σ, J, η)
energy(σ::Vector, h::Vector) = dot(h, σ)
energy(σ::Vector, ig::IsingGraph) = energy(σ, couplings(ig)) + energy(σ, biases(ig))

function brute_force(ig::IsingGraph; sorted=true, num_states::Int=1)
    if nv(ig) == 0 return Spectrum(zeros(1), Vector{Vector{Int}}[]) end
    ig_rank = rank_vec(ig)
    num_states = min(num_states, prod(ig_rank))

    σ = collect.(all_states(ig_rank))
    energies = energy.(σ, Ref(ig))
    if sorted
        perm = partialsortperm(vec(energies), 1:num_states)
        return Spectrum(energies[perm], σ[perm])
    else
        return Spectrum(energies[begin:num_states], σ[begin:num_states])
    end
end

function full_spectrum(ig::IsingGraph; num_states::Int=1)
    brute_force(ig, sorted=false, num_states=num_states)
end

struct Spectrum
    energies::Vector{Float64}
    states::Vector{Vector{Int}}
end

function inter_cluster_energy(cl1_states, J::Matrix, cl2_states)
    hcat(collect.(cl1_states)...)' * J * hcat(collect.(cl2_states)...)
end
