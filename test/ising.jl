using CSV
using LinearAlgebra
using LabelledGraphs

@testset "Ising graph cannot be created" begin
    @testset "if input instance contains duplicate edges" begin
        for T ∈ [Float16, Float32, Float64]
            @test_throws ArgumentError ising_graph(
                T,
                Dict((1, 1) => 2.0, (1, 2) => 0.5, (2, 1) => -1.0),
            )
        end
    end
end

for T ∈ [Float16, Float32, Float64]
    for (instance, source) ∈ (
        ("$(@__DIR__)/instances/example.txt", "file"),
        (
            Dict{Tuple{Int,Int},T}(
                (1, 1) => 0.1,
                (2, 2) => 0.5,
                (1, 4) => -2.0,
                (4, 2) => 1.0,
                (1, 2) => -0.3,
            ),
            "array",
        ),
    )
        @testset "Ising graph created from $(source)" begin
            expected_num_vertices = 3
            expected_biases = [T(1 / 10), T(1 / 2), T(0)]
            expected_couplings = Dict(
                LabelledEdge(1, 2) => -T(3 / 10),
                LabelledEdge(1, 4) => -T(2),
                LabelledEdge(2, 4) => T(1),
            )
            expected_J_matrix =
                [[T(0) -T(3 / 10) -T(2)]; [T(0) T(0) T(1)]; [T(0) T(0) T(0)]]

            ig = ising_graph(T, instance)
            @test eltype(ig) == T
            @testset "contains the same number vertices as original instance" begin
                @test nv(ig) == expected_num_vertices
            end
            @testset "has collection of edges comprising all interactions from instance" begin
                # This test uses the fact that edges iterates in the lex ordering.
                @test collect(edges(ig)) ==
                      [LabelledEdge(e...) for e ∈ [(1, 2), (1, 4), (2, 4)]]
            end
            @testset "stores biases as property of vertices" begin
                @test biases(ig) == expected_biases
            end
            @testset "stores couplings both as property of edges and its own property" begin
                @test couplings(ig) == expected_J_matrix
            end
            @testset "has default rank stored for each active vertex" begin
                @test get_prop(ig, :rank) == Dict(1 => 2, 2 => 2, 4 => 2)
            end
        end
    end
end

@testset "Ising graph created with additional parameters" begin
    expected_biases = [-0.1, -0.5, 0.0]
    expected_couplings = Dict(Edge(1, 2) => 0.3, Edge(1, 3) => 2.0, Edge(2, 3) => -1.0)
    expected_couplings = Dict(Edge(1, 2) => 0.3, Edge(1, 3) => 2.0, Edge(2, 3) => -1.0)
    expected_J_matrix = [
        [0 0.3 2.0]
        [0 0 -1.0]
        [0 0 0]
    ]

    ig = ising_graph(
        "$(@__DIR__)/instances/example.txt",
        scale = -1,
        rank_override = Dict(1 => 3, 4 => 4),
    )

    @testset "has rank overriden by rank_override dict" begin
        # TODO: update default value of 2 once original implementation
        # is also updated.
        @test get_prop(ig, :rank) == Dict(1 => 3, 2 => 2, 4 => 4)
    end

    @testset "has coefficients multiplied by given sign" begin
        @test biases(ig) == expected_biases
        @test couplings(ig) == expected_J_matrix
    end
end

@testset "Ising model is correct" begin
    L = 4
    N = L^2
    instance = "$(@__DIR__)/instances/$(N)_001.txt"
    ig = ising_graph(instance)

    @test nv(ig) == N
    for i ∈ 1:N
        @test has_vertex(ig, i)
    end

    A = adjacency_matrix(ig)
    B = zeros(Int, N, N)
    for i ∈ 1:N
        nbrs = SpinGlassNetworks.unique_neighbors(ig, i)
        for j ∈ nbrs
            B[i, j] = 1
        end
    end
    @test B + B' == A

    @testset "Reading from Dict" begin
        instance_dict = Dict()
        ising = CSV.File(instance, types = [Int, Int, Float64], header = 0, comment = "#")
        ising = CSV.File(instance, types = [Int, Int, Float64], header = 0, comment = "#")

        for (i, j, v) ∈ ising
            push!(instance_dict, (i, j) => v)
        end

        ig = ising_graph(instance)
        ig_dict = ising_graph(instance_dict)

        @test nv(ig_dict) == nv(ig)
        @test collect(edges(ig)) == collect(edges(ig_dict))
    end

    @testset "Ground state energy for pathological instance " begin
        m = 3
        n = 4
        t = 3

        β = 1

        instance = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_$(t).txt"
        ising = CSV.File(instance, types = [Int, Int, Float64], header = 0, comment = "#")
        ig = ising_graph(instance)

        conf = [
            [-1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1],
            [-1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1],
            [-1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1],
            [-1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1],
        ]
        eng = energy(conf, ig)
        couplings = Dict()
        for (i, j, v) ∈ ising
            push!(couplings, (i, j) => v)
        end

        cedges = Dict()
        push!(cedges, (1, 2) => [(1, 4), (1, 5), (1, 6)])
        push!(cedges, (1, 5) => [(1, 13)])
        push!(cedges, (2, 3) => [(4, 7), (5, 7), (6, 8), (6, 9)])
        push!(cedges, (2, 6) => [(6, 16), (6, 18), (5, 16)])
        push!(cedges, (5, 6) => [(13, 16), (13, 18)])
        push!(cedges, (6, 10) => [(18, 28)])
        push!(
            cedges,
            (10, 11) => [
                (28, 31),
                (28, 32),
                (28, 33),
                (29, 31),
                (29, 32),
                (29, 33),
                (30, 31),
                (30, 32),
                (30, 33),
            ],
        )

        push!(cedges, (2, 2) => [(4, 5), (4, 6), (5, 6), (6, 6)])
        push!(cedges, (3, 3) => [(7, 8), (7, 9)])
        push!(cedges, (6, 6) => [(16, 18), (16, 16)])
        push!(cedges, (10, 10) => [(28, 29), (28, 30), (29, 30)])

        config = Dict()
        push!(config, 1 => [-1, -1, -1, -1])
        push!(config, 2 => [0, 0, 0, 0])
        push!(config, 3 => [0, 0, 0, 0])
        push!(config, 4 => [1, 1, 1, 1])
        push!(config, 5 => [1, 1, 1, 1])
        push!(config, 6 => [-1, -1, -1, -1])
        push!(config, 7 => [-1, -1, -1, -1])
        push!(config, 8 => [-1, -1, 1, 1])
        push!(config, 9 => [1, 1, 1, 1])
        push!(config, 10 => [0, 0, 0, 0])
        push!(config, 11 => [0, 0, 0, 0])
        push!(config, 12 => [0, 0, 0, 0])
        push!(config, 13 => [1, 1, 1, 1])
        push!(config, 14 => [0, 0, 0, 0])
        push!(config, 15 => [0, 0, 0, 0])
        push!(config, 16 => [1, 1, 1, 1])
        push!(config, 17 => [0, 0, 0, 0])
        push!(config, 18 => [-1, -1, -1, -1])
        push!(config, 19 => [0, 0, 0, 0])
        push!(config, 20 => [0, 0, 0, 0])
        push!(config, 21 => [0, 0, 0, 0])
        push!(config, 22 => [0, 0, 0, 0])
        push!(config, 23 => [0, 0, 0, 0])
        push!(config, 24 => [0, 0, 0, 0])
        push!(config, 25 => [0, 0, 0, 0])
        push!(config, 26 => [0, 0, 0, 0])
        push!(config, 27 => [0, 0, 0, 0])
        push!(config, 28 => [1, 1, 1, 1])
        push!(config, 29 => [1, 1, 1, 1])
        push!(config, 30 => [-1, -1, -1, -1])
        push!(config, 31 => [1, 1, 1, 1])
        push!(config, 32 => [-1, -1, -1, -1])
        push!(config, 33 => [1, -1, 1, -1])
        push!(config, 34 => [0, 0, 0, 0])
        push!(config, 35 => [0, 0, 0, 0])
        push!(config, 36 => [0, 0, 0, 0])

        num_config = length(config[1])
        exact_energy = _energy(config, couplings, cedges, num_config)

        low_energies = [
            -16.4,
            -16.4,
            -16.4,
            -16.4,
            -16.1,
            -16.1,
            -16.1,
            -16.1,
            -15.9,
            -15.9,
            -15.9,
            -15.9,
            -15.9,
            -15.9,
            -15.6,
            -15.6,
            -15.6,
            -15.6,
            -15.6,
            -15.6,
            -15.4,
            -15.4,
        ]
        for i ∈ 1:num_config
            @test exact_energy[i] == low_energies[i] == eng[i]
        end
    end
end

@testset "Pruning" begin
    @testset "No vertices of degree zero" begin
        instance = Dict(
            (1, 1) => 0.1,
            (2, 2) => 0.5,
            (1, 4) => -2.0,
            (4, 2) => 1.0,
            (1, 2) => -0.3,
        )
        ig = ising_graph(instance)
        ng = prune(ig)
        @test nv(ig) == nv(ng)
    end

    @testset "All vertices of degree zero with no local fields" begin
        instance = Dict((1, 1) => 0.0, (2, 2) => 0.0)
        ig = ising_graph(instance)
        ng = prune(ig)
        @test nv(ng) == 0
    end

    @testset "All vertices of degree zero, but with local fields" begin
        instance = Dict((1, 1) => 0.1, (2, 2) => 0.5)
        ig = ising_graph(instance)
        ng = prune(ig)
        @test nv(ng) == 2
    end

    @testset "Some vertices of degree zero, but nonzero field" begin
        instance = Dict(
            (1, 1) => 0.1,
            (2, 2) => 0.5,
            (1, 4) => -2.0,
            (4, 2) => 1.0,
            (1, 2) => -0.3,
            (5, 5) => 0.1,
        )
        ig = ising_graph(instance)
        ng = prune(ig)
        @test nv(ng) == nv(ig)
        @test vertices(ng) == collect(1:nv(ng))
    end

    @testset "Some vertices of degree zero and zero field" begin
        instance = Dict(
            (1, 1) => 0.1,
            (2, 2) => 0.5,
            (1, 4) => -2.0,
            (4, 2) => 1.0,
            (1, 2) => -0.3,
            (5, 5) => 0.0,
        )
        ig = ising_graph(instance)
        ng = prune(ig)
        @test nv(ng) == nv(ig) - 1
        @test vertices(ng) == collect(1:nv(ng))
    end
end
