# vector_of_vectors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# matrix = transpose(hcat(vector_of_vectors...))
# println(matrix)

A = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
println(typeof(Array(A)))
println(matrix_to_integers(A))