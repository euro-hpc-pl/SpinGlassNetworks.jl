
using SpinGlassNetworks, LightGraphs
using CUDA, LinearAlgebra
function bench(instance::String)
    m = 2
    n = 2
    t = 24

    ig = ising_graph(instance)
    cl = split_into_clusters(ig, super_square_lattice((m, n, t)))
    @time brute_force(cl[1, 1], num_states=100)
    nothing
end

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
    end
    # @time @cuda threads=1024 blocks=4 kernel(J, energies)
    nothing
end

# bench("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt");
bench2("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt");
bench2("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt");
