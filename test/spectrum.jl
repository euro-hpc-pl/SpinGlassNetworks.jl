@testset "Factor graph builds on pathological instance" begin
    m = 3
    n = 4
    t = 3

    β = 1.

    L = n * m * t
    max_cl_states = 20

    instance = "$(@__DIR__)/instances/pathological/cross_$(m)_$(n)_dd.txt"
    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        max_cl_states,
        spectrum=brute_force,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    @time sp = brute_force(ig; sorted=true, num_states=states)

end
