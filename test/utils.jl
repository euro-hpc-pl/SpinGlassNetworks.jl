
# It is posible that more thorough tests will be needed

@testset verbose = true "Renumeration works like in python" begin
    #instance = "$(@__DIR__)/instances/pegasus/2x2x3.txt"

    @testset "linear_to_pegasus" begin

        python_P2_ltp = Dict(0 => (0, 0, 0, 0), 11 => (0, 0, 11, 0),
                        18 => (0, 1, 6, 0), 23 => (0, 1, 11, 0))

        python_P4_ltp = Dict(0 => (0, 0, 0, 0), 86 => (0, 2, 4, 2),
                        112 => (0, 3, 1, 1), 215 => (1, 1, 11, 2))

        python_P8_ltp = Dict(0 => (0, 0, 0, 0), 253 => (0, 3, 0, 1),
                        860 => (1, 2, 2, 6),  1175=> (1, 5, 11, 6))

        python_P16_ltp = Dict(0 => (0, 0, 0, 0), 2530 => (0, 14, 0, 10),
                        4860 => (1, 11, 0, 0), 5399 => (1, 13, 11, 14))

        for i in keys(python_P2_ltp)
            @test linear_to_pegasus(2, i) == python_P2_ltp[i]
        end

        for i in keys(python_P4_ltp)
            @test linear_to_pegasus(4, i) == python_P4_ltp[i]
        end

        for i in keys(python_P8_ltp)
            @test linear_to_pegasus(8, i) == python_P8_ltp[i]
        end

        for i in keys(python_P16_ltp)
            @test linear_to_pegasus(16, i) == python_P16_ltp[i]
        end
    end

    @testset "pegasus_to_nice" begin

        python_ptn = Dict((0,0,4,0) => (0, 0, 0, 0, 0), (0, 3, 1, 1) => (2, 1, 2, 0, 1),
                    (1, 5, 11, 6) => (2, 5, 6, 1, 3), (1, 14, 0, 10) => (1, 13, 10, 1, 0))

        for p in keys(python_ptn)
            @test pegasus_to_nice(p) == python_ptn[p]
        end
    end

    @testset "linear_to_nice" begin
        @test linear_to_nice(16, 111) == (0, 6, 0, 0, 3)
    end

    @testset "nice_to_dattani" begin
        @test nice_to_dattani((2, 5, 6, 1, 3)) == (6, 5, 2, 1, 3)
    end

    @testset "dattani_to_linear" begin
        python_P16_dtl = Dict((12, 7, 1, 1, 0) => 4501, (6, 10, 1, 0, 0) => 2409,
                    (3, 2, 2, 0, 1) => 1146, (9, 0, 2, 0, 3)=> 3260, (10, 3, 1, 0, 2) => 3683)
        for i in keys(python_P16_dtl)
            @test dattani_to_linear(16, i) == python_P16_dtl[i]
        end
    end
end

# This test takes long time.

# @testset verbose = true "Renumerated instances generate correct factor graph" begin
#     instance_dir = "$(@__DIR__)/instances/pegasus/"
#     instances = ["P2", "P4", "P8", "P16"]
#     size = [2,4,8,16]

#     @testset "$instance" for (i, instance) ∈ enumerate(instances)

#         instance = instance * ".txt"
#         s = size[i]-1
#         m, n, t = s, s, 24
#         max_cl_states = 2

#         ig = ising_graph(joinpath(instance_dir, instance))
#         cl_h = clustered_hamiltonian(
#             ig,
#             max_cl_states,
#             spectrum = brute_force,
#             cluster_assignment_rule=super_square_lattice((m, n, t))
#         )

#         @test nv(cl_h) == s^2

#         if s > 1
#             for l in 1:s-1, k in 2:s
#                 @test has_edge(cl_h, (l,k), (l+1,k-1))
#             end
#         end
#     end
# end
