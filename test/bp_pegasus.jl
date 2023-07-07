"""
Instance below looks like this:

1 -- 2



"""
function create_larger_example_factor_graph_tree()
   instance = Dict(
      (1, 1) => 0.5,
      (2, 2) => 0.25,
      # (3, 3) => 0.3,
      # (4, 4) => 0.1,
      # (5, 5) => 0.1,
      # (6, 6) => -2.0,
      (1, 2) => -1.0,
      # (1, 3) => 1.0
      # (2, 4) => 1.0,
      # (3, 4) => 1.0,
      # (1, 5) => 0.5,
      # (2, 5) => 0.5,
      # (2, 6) => 0.5,
      # (5, 6) => -0.3,
   )

   ig = ising_graph(instance)

   assignment_rule = Dict(
      1 => (1, 1, 1),
      2 => (1, 2, 1),
      # 3 => (2, 1, 1)
      # 4 => (1, 2, 2),
      # 5 => (2, 1, 1),
      # 6 => (2, 1, 2),
   )

   fg = factor_graph(
      ig,
      Dict{NTuple{3, Int}, Int}(),
      spectrum = full_spectrum,
      cluster_assignment_rule = assignment_rule,
   )

   ig, fg
end

function exact_cond_prob(factor_graph::LabelledGraph{S, T}, beta, target_state::Dict) where {S, T}  # TODO: Not going to work without PoolOfProjectors
   ver = vertices(factor_graph)
   rank = cluster_size.(Ref(factor_graph), ver)
   states = [Dict(ver .=> σ) for σ ∈ Iterators.product([1:r for r ∈ rank]...)]
   energies = SpinGlassNetworks.energy.(Ref(factor_graph), states)
   prob = exp.(-beta .* energies)
   prob ./= sum(prob)
   sum(prob[findall([all(s[k] == v for (k, v) ∈ target_state) for s ∈ states])])
end

ig, fg = create_larger_example_factor_graph_tree()
beta = 1
beliefs = belief_propagation(fg, beta; iter=100)
println(beliefs)

println(exact_cond_prob(fg, beta, Dict((1, 1, 1) => 1)))
