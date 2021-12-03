
using SpinGlassNetworks

function bench(instance::String)
    m = 2
    n = 2
    t = 3
    max_cl_states = 100

    @time ig = ising_graph(instance)

    @time fg = factor_graph(
        ig,
        max_cl_states,
        spectrum=brute_force,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )
end

bench("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt")
