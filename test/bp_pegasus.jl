"""
Instance below looks like this:

1 -- 2
|
3 -- 4

"""
function create_larger_example_factor_graph_tree_basic()
   instance = Dict(
      (1, 1) => 0.5,
      (2, 2) => 0.25,
      (3, 3) => 0.3,
      (4, 4) => 0.1,
      (1, 2) => -1.0,
      (1, 3) => 1.0,
      (3, 4) => 1.0
   )

   ig = ising_graph(instance)

   assignment_rule = Dict(
      1 => (1, 1, 1),
      2 => (1, 2, 1),
      3 => (2, 1, 1),
      4 => (2, 2, 2)
   )

   fg = factor_graph(
      ig,
      Dict{NTuple{3, Int}, Int}(),
      spectrum = full_spectrum,
      cluster_assignment_rule = assignment_rule,
   )

   ig, fg
end

"""
Instance below looks like this:

1 -- 2 -- 3
|
4 -- 5 -- 6
| 
7 -- 8 -- 9
"""
function create_larger_example_factor_graph_tree()
   instance = Dict(
      (1, 1) => 0.5,
      (2, 2) => 0.25,
      (3, 3) => 0.3,
      (4, 4) => 0.1,
      (5, 5) => -0.1,
      (6, 6) => 0.1,
      (7, 7) => 0.0,
      (8, 8) => 0.1,
      (9, 9) => 0.01,
      (1, 2) => -1.0,
      (2, 3) => 1.0,
      (1, 4) => 1.0,
      (4, 5) => 1.0,
      (5, 6) => 1.0,
      (4, 7) => 1.0,
      (7, 8) => 1.0,
      (8, 9) => 1.0
   )

   ig = ising_graph(instance)

   assignment_rule = Dict(
      1 => (1, 1, 1),
      2 => (1, 2, 1),
      3 => (1, 3, 1),
      4 => (2, 1, 1),
      5 => (2, 2, 1),
      6 => (2, 3, 1),
      7 => (3, 1, 1),
      8 => (3, 2, 1),
      9 => (3, 3, 1)
   )

   fg = factor_graph(
      ig,
      Dict{NTuple{3, Int}, Int}(),
      spectrum = full_spectrum,
      cluster_assignment_rule = assignment_rule,
   )

   ig, fg
end

"""
Instance below looks like this:

1 -- 2 -- 3
|    |
4    5

"""
function create_larger_example_factor_graph_tree_pathological()
   instance = Dict(
      (1, 1) => 0.5,
      (2, 2) => 0.25,
      (3, 3) => 0.3,
      (4, 4) => 0.1,
      (5, 5) => -0.1,
      (6, 6) => 0.1,
      (7, 7) => 0.0,
      (8, 8) => 0.1,
      (1, 2) => -1.0,
      (1, 3) => 1.0,
      (3, 4) => 1.0,
      (3, 5) => 1.0,
      (5, 6) => 1.0,
      (2, 7) => 0.5,
      (3, 7) => 1.0,
      (5, 8) => 1.0,
      
   )

   ig = ising_graph(instance)

   assignment_rule = Dict(
      1 => (1, 1),
      2 => (1, 1),
      3 => (1, 1),
      4 => (1, 2),
      5 => (1, 2),
      6 => (1, 3),
      7 => (2, 1),
      8 => (2, 2)
   )

   fg = factor_graph(
      ig,
      Dict{NTuple{2, Int}, Int}(),
      spectrum = full_spectrum,
      cluster_assignment_rule = assignment_rule,
   )

   ig, fg
end

"""
Instance below looks like this:

1 -- 2
|
3 -- 4

"""
function create_larger_example_factor_graph_tree_2site()
   instance = Dict(
      (1, 1) => 0.5,
      (2, 2) => 0.25,
      (3, 3) => 0.3,
      (4, 4) => 0.1,
      (5, 5) => 0.1,
      (6, 6) => 0.1,
      (7, 7) => 0.1,
      (8, 8) => 0.1,
      (1, 2) => -0.00001,
      (1, 3) => 1.0,
      (3, 4) => 1.0,
      (2, 5) => 1.0,
      (3, 4) => 1.0,
      (6, 7) => 1.0,
      (6, 8) => 1.0,
      (6, 2) => 1.0
   )

   ig = ising_graph(instance)

   assignment_rule = Dict(
      1 => (1, 1, 1),
      2 => (1, 1, 2),
      3 => (1, 2, 1),
      4 => (1, 2, 2),
      5 => (2, 1, 1),
      6 => (2, 1, 2),
      7 => (2, 2, 1),
      8 => (2, 2, 2)
   )

   fg = factor_graph(
      ig,
      Dict{NTuple{3, Int}, Int}(),
      spectrum = full_spectrum,
      cluster_assignment_rule = assignment_rule,
   )

   ig, fg
end



@testset "Belief propagation basic" begin
   ig, fg = create_larger_example_factor_graph_tree_basic()
   beta = 1
   iter = 100
   beliefs = belief_propagation(fg, beta; iter=iter)
   exact_marginal = Dict()
   for k in keys(beliefs)
      push!(exact_marginal, k => [exact_cond_prob(fg, beta, Dict(k => a)) for a in 1:length(beliefs[k])])
   end
   for v in keys(beliefs)
      temp = -log.(exact_marginal[v])./beta
      @test beliefs[v] ≈ temp .- minimum(temp)
   end
end
@testset "Belief propagation " begin
   ig, fg = create_larger_example_factor_graph_tree()
   beta = 1
   iter = 100
   beliefs = belief_propagation(fg, beta; iter=iter)
   exact_marginal = Dict()
   for k in keys(beliefs)
      push!(exact_marginal, k => [exact_cond_prob(fg, beta, Dict(k => a)) for a in 1:length(beliefs[k])])
   end
   for v in keys(beliefs)
      temp = -log.(exact_marginal[v])./beta
      @test beliefs[v] ≈ temp .- minimum(temp)   end
end
@testset "Belief propagation pathological" begin
   ig, fg = create_larger_example_factor_graph_tree_pathological()
   beta = 1
   iter = 100
   beliefs = belief_propagation(fg, beta; iter=iter)
   exact_marginal = Dict()
   for k in keys(beliefs)
      push!(exact_marginal, k => [exact_cond_prob(fg, beta, Dict(k => a)) for a in 1:length(beliefs[k])])
   end
   for v in keys(beliefs)
      temp = -log.(exact_marginal[v])./beta
      @test beliefs[v] ≈ temp .- minimum(temp)   end
end

# @testset "Belief propagation 2site" begin
#    ig, fg = create_larger_example_factor_graph_tree_2site()
#    beta = 1
#    tol = 1e-12
#    iter = 100
#    beliefs, messages_av = belief_propagation_old(fg, beta; iter=iter, tol=tol, output_message=true)
#    exact_marginal = Dict()
#    for k in keys(beliefs)
#       push!(exact_marginal, k => [exact_cond_prob(fg, beta, Dict(k => a)) for a in 1:length(beliefs[k])])
#    end
#    for v in keys(beliefs)
#       @test beliefs[v] ≈ exact_marginal[v]
#    end

#    for v in vertices(fg)
#       println("v ", v)
#       i, j, _ = v
#       n1, n2 = length(beliefs[(i, j, 1)]), length(beliefs[(i, j, 2)])
#       exact = [exact_cond_prob(fg, beta, Dict((i, j, 1) => k1, (i, j, 2) => k2)) for k1 in 1:n1, k2 in 1:n2]
#       println("exact ", exact)
#       belief = beliefs_2site(fg, i, j, messages_av, beta)
#       println("belief ", belief)

#       # @test beliefs_2site(fg, i, j, messages_av, beta) ≈ exact
#    end

# end