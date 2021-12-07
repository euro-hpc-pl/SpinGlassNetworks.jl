
using SpinGlassNetworks

function bench(instance::String, max_states::Int=100)
    m = 2
    n = 2
    t = 24

    println("Threads: ", Threads.nthreads())

    ig = ising_graph(instance)
    cl = split_into_clusters(ig, super_square_lattice((m, n, t)))
    @time sp = brute_force(cl[1, 1], num_states=max_states)
    return
end

bench("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt")
