export
    all_states,
    local_basis,
    gibbs_tensor,
    brute_force,
    full_spectrum,
    Spectrum,
    idx,
    local_basis,
    energy

@inline idx(σ::Int) = (σ == -1) ? 1 : σ + 1
@inline local_basis(d::Int) = union(-1, 1:d-1)
all_states(rank::Union{Vector, NTuple}) = Iterators.product(local_basis.(rank)...)

const State = Vector{Int}
struct Spectrum
    energies::Vector{<:Real}
    states::AbstractArray{State}
end

function energy(σ::AbstractArray{State}, ig::IsingGraph)
    J, h = couplings(ig), biases(ig)
    dot.(σ, Ref(J), σ) + dot.(Ref(h), σ)
end

function Spectrum(ig::IsingGraph)
    L = nv(ig)
    N = 2^L

    energies = zeros(Float64, N)
    states = Vector{State}(undef, N)

    J, h = couplings(ig), biases(ig)
    Threads.@threads for i = 0:N-1
        σ = 2 .* digits(i, base=2, pad=L) .- 1
        @inbounds energies[i+1] = dot(σ, J, σ) + dot(h, σ)
        @inbounds states[i+1] = σ
    end
    Spectrum(energies, states)
end

function gibbs_tensor(ig::IsingGraph, β::Real=1.0)
    σ = collect.(all_states(rank_vec(ig)))
    ρ = exp.(-β .* energy(σ, ig))
    ρ ./ sum(ρ)
end

function brute_force(ig::IsingGraph, s::Symbol=:CPU; num_states::Int=1)
    brute_force(ig, Val(s); num_states)
end

function brute_force(ig::IsingGraph, ::Val{:CPU}; num_states::Int=1)
    L = nv(ig)
    if L == 0 return Spectrum(zeros(1), Vector{Vector{Int}}[]) end
    sp = Spectrum(ig)
    num_states = min(num_states, prod(rank_vec(ig)))
    idx = partialsortperm(vec(sp.energies), 1:num_states)
    Spectrum(sp.energies[idx], sp.states[idx])
end

function full_spectrum(ig::IsingGraph; num_states::Int=1)
    if nv(ig) == 0 return Spectrum(zeros(1), Vector{Vector{Int}}[]) end
    ig_rank = rank_vec(ig)
    num_states = min(num_states, prod(ig_rank))
    σ = collect.(all_states(ig_rank))
    energies = energy(σ, ig)
    Spectrum(energies[begin:num_states], σ[begin:num_states])
end

function inter_cluster_energy(
    cl1_states::Vector{State},
    J::Matrix{<:Real},
    cl2_states::Vector{State}
)
    hcat(collect.(cl1_states)...)' * J * hcat(collect.(cl2_states)...)
end
