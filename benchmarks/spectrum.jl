
using SpinGlassNetworks

function bench(instance::String)
    m = 2
    n = 2
    t = 24

    ig = ising_graph(instance)
    cl = split_into_clusters(ig, super_square_lattice((m, n, t)))
    @time sp = brute_force(cl[1, 1], num_states=100)
end

bench("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt");
