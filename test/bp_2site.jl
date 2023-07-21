using TensorCast

"""
Instance below looks like this:

12 -- 34
|
56 -- 78

"""
function create_larger_example_factor_graph_tree_2site()
   instance = Dict(
      (1, 1) =>  0.50,
      (2, 2) => -0.25,
      (3, 3) =>  0.30,
      (4, 4) => -0.10,
      (5, 5) =>  0.10,
      (6, 6) =>  0.20,
      (7, 7) => -0.40,
      (8, 8) =>  0.50,
      (1, 2) => -0.01,
      (1, 3) =>  1.00,
      (3, 4) =>  1.10,
      (2, 5) =>  0.77,
      (6, 7) => -1.43,
      (6, 8) =>  0.21,
      (6, 2) =>  1.00
   )

   ig = ising_graph(instance)

   assignment_rule1 = Dict(
      1 => (1, 1),
      2 => (1, 1),
      3 => (1, 2),
      4 => (1, 2),
      5 => (2, 1),
      6 => (2, 1),
      7 => (2, 2),
      8 => (2, 2)
   )

   assignment_rule2 = Dict(
      1 => (1, 1, 1),
      2 => (1, 1, 2),
      3 => (1, 2, 1),
      4 => (1, 2, 2),
      5 => (2, 1, 1),
      6 => (2, 1, 2),
      7 => (2, 2, 1),
      8 => (2, 2, 2)
   )

   fg1 = factor_graph(
      ig,
      Dict{NTuple{2, Int}, Int}(),
      spectrum = full_spectrum,
      cluster_assignment_rule = assignment_rule1,
   )

   fg2 = factor_graph(
      ig,
      Dict{NTuple{3, Int}, Int}(),
      spectrum = full_spectrum,
      cluster_assignment_rule = assignment_rule2,
   )

   ig, fg1, fg2
end

"""
Instance below looks like this:

1 -- 2 -- 3
     |
     4
     |
     5

"""
function create_larger_example_factor_graph_tree_2site_pathological()
   instance = Dict(
      (1, 1) => -0.50,
      (2, 2) =>  0.25,
      (3, 3) =>  0.31,
      (4, 4) =>  0.77,
      (5, 5) => -0.15,
      (6, 6) =>  0.21,
      (7, 7) => -0.10,
      (8, 8) =>  0.13,
      (9, 9) =>  0.01,
      (10, 10) => 0.5,
      (1, 2) => -0.01,
      (1, 3) =>  0.79,
      (2, 5) => -0.93,
      (5, 6) =>  0.81,
      (5, 9) => -0.17,
      (3, 4) =>  0.71,
      (4, 6) => -0.43,
      (5, 8) =>  0.75,
      (7, 10)=> -1.03,
      (8, 9) =>  0.65,
      (9, 10)=> -0.57
   )

   ig = ising_graph(instance)

   assignment_rule1 = Dict(
      1 => (1, 1),
      2 => (1, 1),
      3 => (1, 2),
      4 => (1, 2),
      5 => (1, 2),
      6 => (1, 3),
      7 => (2, 2),
      8 => (2, 2),
      9 => (2, 2),
      10 => (3, 2)
   )

   assignment_rule2 = Dict(
      1 => (1, 1, 1),
      2 => (1, 1, 2),
      3 => (1, 2, 1),
      4 => (1, 2, 1),
      5 => (1, 2, 2),
      6 => (1, 3, 2),
      7 => (2, 2, 1),
      8 => (2, 2, 1),
      9 => (2, 2, 2),
      10 => (3, 2, 2)
   )

   fg1 = factor_graph(
      ig,
      Dict{NTuple{2, Int}, Int}(),
      spectrum = full_spectrum,
      cluster_assignment_rule = assignment_rule1,
   )

   fg2 = factor_graph(
      ig,
      Dict{NTuple{3, Int}, Int}(),
      spectrum = full_spectrum,
      cluster_assignment_rule = assignment_rule2,
   )

   ig, fg1, fg2
end


@testset "Belief propagation 2site" begin

    for (ig, fg1, fg2) ∈ [create_larger_example_factor_graph_tree_2site(),
                          create_larger_example_factor_graph_tree_2site_pathological()]
      for beta ∈ [0.6, 1.1]
         tol = 1e-12
         iter = 16
         num_states=10

         new_fg1 = factor_graph_2site(fg2, beta)

         @test vertices(new_fg1) == vertices(fg1)
         @test edges(new_fg1) == edges(fg1)
         for e ∈ vertices(new_fg1)
            @test get_prop(new_fg1, e, :spectrum).energies ≈ get_prop(fg1, e, :spectrum).energies
         end
         for e ∈ edges(new_fg1)
            E = get_prop(new_fg1, src(e), dst(e), :en)
            @cast E[(l1, l2), (r1, r2)] := E.e11[l1, r1] + E.e21[l2, r1] + E.e12[l1, r2] + E.e22[l2, r2]
            @test E == get_prop(fg1, src(e), dst(e), :en)
         end
         for e ∈ edges(new_fg1)
            @test get_prop(new_fg1, src(e), dst(e), :pl) == get_prop(fg1, src(e), dst(e), :pl)
            @test get_prop(new_fg1, src(e), dst(e), :pr) == get_prop(fg1, src(e), dst(e), :pr)
         end

         beliefs = belief_propagation(new_fg1, beta; iter=iter, tol=tol, output_message=false)

         exact_marginal = Dict()
         for k in keys(beliefs)
            temp = -1/beta .* log.([exact_cond_prob(fg1, beta, Dict(k => a)) for a in 1:length(beliefs[k])])
            push!(exact_marginal, k => temp .- minimum(temp))
         end
         for v in keys(beliefs)
            @test beliefs[v] ≈ exact_marginal[v]
         end

         truncate_factor_graph_2site_BP(fg2, num_states; beta=beta, iter=iter)
      end
   end
 end
