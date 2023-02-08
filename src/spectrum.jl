export all_states, local_basis
export gibbs_tensor
export brute_force, full_spectrum, energy
export Spectrum

"""
$(TYPEDSIGNATURES)

Calculates Gibbs state of a classical Ising Hamiltonian

# Details

Calculates matrix elements (probabilities) of \$\\rho\$
```math
\$\\bra{\\σ}\\rho\\ket{\\sigma}\$
```
for all possible configurations \$\\σ\$.
"""
function gibbs_tensor(ig::IsingGraph, β = Float64 = 1.0)
    states = collect.(all_states(rank_vec(ig)))
    ρ = exp.(-β .* energy.(states, Ref(ig)))
    ρ ./ sum(ρ)
end


"""
$(TYPEDSIGNATURES)

Calculate the Ising energy
```math
E = -\\sum_<i,j> s_i J_{ij} * s_j - \\sum_j h_i s_j.
```
"""
energy(σ::Vector, J::Matrix, η::Vector = σ) = dot(σ, J, η)
energy(σ::Vector, h::Vector) = dot(h, σ)
energy(σ::Vector, ig::IsingGraph) = energy(σ, couplings(ig)) + energy(σ, biases(ig))


"""
$(TYPEDSIGNATURES)

Return the low energy spectrum

# Details

Calculates \$k\$ lowest energy states
together with the coresponding energies
of a classical Ising Hamiltonian
"""

function brute_force(ig::IsingGraph; sorted = true, num_states::Int = 1)
    if nv(ig) == 0
        return Spectrum(zeros(1), [])
    end
    ig_rank = rank_vec(ig)
    num_states = min(num_states, prod(ig_rank))

    σ = collect.(all_states(ig_rank))
    energies = energy.(σ, Ref(ig))
    if sorted
        perm = partialsortperm(vec(energies), 1:num_states)
        return Spectrum(energies[perm], σ[perm])
    else
        return Spectrum(energies[1:num_states], σ[1:num_states])
    end
end

full_spectrum(ig::IsingGraph; num_states::Int = 1) =
    brute_force(ig, sorted = false, num_states = num_states)

struct Spectrum
    energies::Vector{Float64}
    states::Vector{Vector{Int}}
end

# Please don't make the below another energy method.
# There is already so much mess going on :)
function inter_cluster_energy(cl1_states, J::Matrix, cl2_states)
    hcat(collect.(cl1_states)...)' * J * hcat(collect.(cl2_states)...)
end
