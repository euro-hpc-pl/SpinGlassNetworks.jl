using NaturalSort

@testset verbose = true "Numeration generate correct diagonals" begin

    size = [2, 4, 8, 16]

    @testset "$i" for (i, instance) ∈ enumerate(sort(readdir("$(@__DIR__)/instances/pegasus/", join=false), lt = natural))
        println(size[i])
        println(instance)


        #=instance = "$(@__DIR__)/instances/pegasus/001.txt"
        m, n, t = 3, 3, 24
        max_cl_states = 2^2

        ig = ising_graph(instance)
        fg = factor_graph(
            ig,
            max_cl_states,
            spectrum = brute_force, # to use CPU
            cluster_assignment_rule=super_square_lattice((m, n, t))
        )

        for edge in edges(fg)
            println(edge)
        end
        =#
    end
end