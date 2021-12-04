export all_states, local_basis, gibbs_tensor, brute_force
export full_spectrum, Spectrum, idx, local_basis, energy

local_basis(d::Int) = union(-1, 1:d-1)
all_states(rank::Union{Vector, NTuple}) = Iterators.product(local_basis.(rank)...)

struct Spectrum
    energies::Vector{Float64}
    states::Vector{Vector{Int}}
end

function Spectrum(ig::IsingGraph)
    L = nv(ig)
    N = 2^L

    energies = zeros(Float64, N)
    states = Vector{Vector{Int}}(undef, N)

    J, h = couplings(ig), biases(ig)
    Threads.@threads for i = 0:N-1
        σ = 2 .* digits(i, base=2, pad=L) .- 1
        @inbounds energies[i+1] = dot(σ, J, σ) + dot(h, σ)
        @inbounds states[i+1] = σ
    end
    Spectrum(energies, states)
end

#=
function _kernel(
    energies, states, J, h
)
    L = size(J, 1)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    σ = 2 .* digits(i, base=2, pad=L) .- 1
    @inbounds energies[i] = dot(σ, J, σ) + dot(h, σ)
    @inbounds states[i] = σ
    nothing
end

function CUDASpectrum(ig::IsingGraph)
    L = nv(ig)
    N = 2^L

    en_d = CUDA.zeros(Float64, N)
    st_d = CUDA.Vector{Vector{Int}}(undef, N)
    J_d, h_d = CUDA.zeros(L, L), CUDA.zeros(L)

    copyto!(couplings(ig), J_d)
    copyto!(biases(ig), h_d)

    nb = ceil(Int, N/256)
    CUDA.@sync begin
        @cuda threads=N blocks=nb _kernel(en_d, st_d, J_d, h_d)
    end
    Spectrum(Array(en_d), Array(st_d))
end
=#

function energy(
    σ::AbstractArray{Vector{Int}}, J::Matrix{<:Real}, h::Vector{<:Real}
)
    dot.(σ, Ref(J), σ) + dot.(Ref(h), σ)
end

function gibbs_tensor(ig::IsingGraph, β::Real=1.0)
    σ = collect.(all_states(rank_vec(ig)))
    ρ = exp.(-β .* energy(σ, couplings(ig), biases(ig)))
    ρ ./ sum(ρ)
end

function brute_force(ig::IsingGraph; num_states::Int=1)
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
    energies = energy(σ, couplings(ig), biases(ig))
    Spectrum(energies[begin:num_states], σ[begin:num_states])
end

function inter_cluster_energy(cl1_states, J::Matrix, cl2_states)
    hcat(collect.(cl1_states)...)' * J * hcat(collect.(cl2_states)...)
end
