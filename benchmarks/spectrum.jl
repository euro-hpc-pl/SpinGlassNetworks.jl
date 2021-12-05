
using SpinGlassNetworks, LightGraphs
using CUDA, LinearAlgebra
using Bits

function bench(instance::String)
    m = 2
    n = 2
    t = 24

    ig = ising_graph(instance)
    cl = split_into_clusters(ig, super_square_lattice((m, n, t)))
    @time sp = brute_force(cl[1, 1], num_states=100)
    sp
end

function my_digits(d::Int, L::Int)
    σ = zeros(Int, L)
    for i=1:L if tstbit(d, i) @inbounds σ[i] = 1 end end
    σ
end

function kernel(J, energies, σ)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    L = size(σ, 1)

    for i=1:L if tstbit(idx, i) @inbounds σ[i] = 1 end end

    for k=1:L
        @inbounds energies[idx] += J[k, k] * σ[k]
        for l=1:L @inbounds energies[idx] += σ[k] * J[k, l] * σ[l] end # 1 -> (k+1)
    end
    return
end

function bench3(instance::String)
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
        σ = CUDA.zeros(L)
        J_dev = CUDA.CuArray(J)
        @cuda threads=1024 blocks=(2^(L-10)) kernel(J_dev, energies, σ)
        energies_cpu = Array(energies)
        σ_cpu = Array(σ)
        println(σ[1:5])
        # perm = sortperm(energies_cpu)
        #sortperm(energies)
    end
    # @time @cuda threads=1024 blocks=4 kernel(J, energies)
    energies_cpu
end

#=
function kernel(J, energies, σ)
    L = size(J, 1)

    i = blockIdx().x
    j = threadIdx().x

    s = (i - 1) * blockDim().x + j
    s1 = copy(s)

    for k=1:L
        @inbounds σ[k, s] = s1%2
        if s1 == 1
            break
        end
        s1 = div(s1, 2)
    end

    for k=1:L
        @inbounds energies[s] += J[k, k] * σ[k, s]
        for l=(k+1):L
            @inbounds energies[s] += J[k, l] * σ[k, s] * σ[l, s]
        end
    end

    return
end

function bench2(instance::String)
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
        σ = CUDA.zeros(L, N)
        J_dev = CUDA.CuArray(J)
        @cuda threads=1024 blocks=(2^(L-10)) kernel(J_dev, energies, σ)
        energies_cpu = Array(energies)
        # perm = sortperm(energies_cpu)
        sortperm(energies)
    end
    # @time @cuda threads=1024 blocks=4 kernel(J, energies)
    energies_cpu
end
=#

sp = bench("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt");
en = bench3("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt");
#bench2("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt");

#minimum(sp.energies) ≈ minimum(en)

println(minimum(sp.energies))
println(en[1:10])


L=100
@assert all(my_digits(i, L) == digits(i, base=2, pad=L) for i=1:L)
