using HDF5
using LightGraphs
using LabelledGraphs
using MetaGraphs
using SpinGlassNetworks


function load_openGM(fname::String, Nx::Integer, Ny::Integer)
    file = h5open(fname, "r")

    file_keys = collect(keys(read(file)))
    data = read(file[file_keys[1]])
    H = collect(Int64, data["header"])
    F = Array{Int64}(data["factors"])
    J = Array{Int64}(data["function-id-16000"]["indices"])
    V = Array{Real}(data["function-id-16000"]["values"])
    N = Array{Int64}(data["numbers-of-states"])

    F = reverse(F)
    factors = Dict()

    while length(F) > 0
        f1 = pop!(F)
        z1 = pop!(F)
        nn = pop!(F)
        n = []

        for _ in 1:nn
            tt = pop!(F)
            ny, nx = divrem(tt, Nx)
            push!(n, ny, nx)
        end

        if length(n) == 4
            if abs(n[1] - n[3]) + abs(n[2] - n[4]) != 1
                throw(Exception("Not nearest neighbour"))
            end
        end

        if length(n) == 2
            if (n[1] >= Ny) || (n[2] >= Nx)
                throw(Exception("Wrong size"))
            end
        end

        factors[tuple(n...)] = f1

        if z1 != 0
            throw(Exception("Something wrong with the expected convention."))
        end
    end

    J = reverse(J)
    functions = Dict()
    ii = -1
    lower = 0

    while length(J) > 0
        ii += 1
        nn = pop!(J)
        n = []

        for _ in 1:nn
            push!(n, pop!(J))
        end

        upper = lower + prod(n)
        functions[ii] = reshape(V[lower + 1:upper], reverse(n)...)'

        lower = upper
    end

    result = Dict("fun" => functions, "fac" => factors, "N" => reshape(N, (Ny, Nx)), "Nx" => Nx, "Ny" => Ny)
    result
end

function energy_rmf()
    
end

function clustered_hamiltonian(fname::String, Nx::Integer = 240, Ny::Integer = 320)
    loaded_rmf = load_openGM(fname, Nx, Ny)
    functions = loaded_rmf["fun"]
    factors = loaded_rmf["fac"]
    N = loaded_rmf["N"]
    println(size(N))
    node_factors = Dict()
    edge_factors = Dict()

    for index in keys(factors)
        if length(index) == 4
            push!(edge_factors, index=>factors[index])
        else
            push!(node_factors, index=>factors[index])
        end
    end
    println(length(node_factors))
    println((0,0) in collect(keys(node_factors)))
    g = grid((Nx,Ny))
    clusters = super_square_lattice((Nx, Ny, 1))
    cl_h = LabelledGraph{MetaDiGraph}(sort(collect(values(clusters))))
    for v ∈ cl_h.labels
        x, y = v
        sp = Spectrum([node_factors[(y-1, x-1)]], Array([collect(1:N[y, x])]), Vector{Int}([]))
        set_props!(cl_h, v, Dict(:cluster => v, :spectrum => sp))
    end

    cl_h
end


x, y = 240, 320
filename = "/home/tsmierzchalski/.julia/dev/SpinGlassNetworks/examples/penguin-small.h5"


cf = clustered_hamiltonian(filename, x, y)
