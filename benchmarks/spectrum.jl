
using SpinGlassNetworks, LightGraphs
using CUDA, LinearAlgebra
using Bits

function bench_cpu(instance::String, max_states::Int=100)
    m = 2
    n = 2
    t = 24

    ig = ising_graph(instance)
    cl = split_into_clusters(ig, super_square_lattice((m, n, t)))
    @time sp = brute_force(cl[1, 1], num_states=max_states)
    sp
end

function kernel(J, energies, σ)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    L = size(σ, 1)

    for i=1:L if tstbit(idx, i) @inbounds σ[i, idx] = 1 end end

    for k=1:L
        @inbounds energies[idx] += J[k, k] * σ[k, idx]
        for l=1:L @inbounds energies[idx] += σ[k, idx] * J[k, l] * σ[l, idx] end # 1 -> (k+1)
    end
    return
end

function bench_gpu(instance::String, max_states::Int=100)
    m = 2
    n = 2
    t = 24

    ig = ising_graph(instance)
    cl = split_into_clusters(ig, super_square_lattice((m, n, t)))
    J = couplings(cl[1, 1]) + Diagonal(biases(cl[1, 1]))

    L = nv(cl[1, 1])
    N = 2^L
    @time begin
        energies = CUDA.zeros(N)
        σ = CUDA.zeros(Int, L, N) .- 1
        J_dev = CUDA.CuArray(J)
        @cuda threads=1024 blocks=(2^(L-10)) kernel(J_dev, energies, σ)
        #sortperm(energies)
        energies_cpu = Array(energies)
        σ_cpu = Array(σ)
        perm = partialsortperm(energies_cpu, 1:max_states)
    end
    Spectrum(energies_cpu[perm], σ_cpu[:, perm])
end


sp_cpu = bench_cpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt");
sp_gpu = bench_gpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt");

sp_cpu.energies ≈ sp_gpu.energies

println(minimum(sp_gpu.energies))
println(minimum(sp_cpu.energies))
