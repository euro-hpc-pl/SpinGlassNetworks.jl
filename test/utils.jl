@testset verbose = true "Renumerated instances generate correct factor graph" begin
    instance_dir = "$(@__DIR__)/instances/pegasus/"
    instances = ["P2", "P4", "P8", "P16"]
    size = [2, 4, 8, 16]

    @testset "$instance" for (i, instance) ∈ enumerate(instances)

        instance = instance * ".txt"
        s = size[i] - 1
        m, n, t = s, s, 24
        max_cl_states = 2

        ig = ising_graph(joinpath(instance_dir, instance))
        fg = factor_graph(
            ig,
            max_cl_states,
            spectrum = brute_force,
            cluster_assignment_rule = super_square_lattice((m, n, t))
        )
        @test nv(fg) == s^2

        if s > 1
            for l in 1:s-1, k in 2:s
                @test has_edge(fg, (l, k), (l+1, k-1))
            end
        end
    end
end
