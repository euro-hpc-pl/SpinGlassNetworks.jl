module SpinGlassNetworks
    using LabelledGraphs
    using LightGraphs
    using MetaGraphs # TODO: remove that
    using CSV
    using DocStringExtensions
    using LinearAlgebra
    using Base.Cartesian
    using CUDA

    import Base.Prehashed

    export unique_neighbors


    unique_neighbors(ig::LabelledGraph, i::Int) = filter(j -> j > i, neighbors(ig, i))

    @generated function unique_dims(A::AbstractArray{T,N}, dim::Integer) where {T,N}
        quote
            1 <= dim <= $N || return copy(A)
            hashes = zeros(UInt, axes(A, dim))

            # Compute hash for each row
            k = 0
            @nloops $N i A d->(if d == dim; k = i_d; end) begin
                @inbounds hashes[k] = hash(hashes[k], hash((@nref $N A i)))
            end

            # Collect index of first row for each hash
            uniquerow = similar(Array{Int}, axes(A, dim))
            firstrow = Dict{Prehashed,Int}()
            for k = axes(A, dim)
                uniquerow[k] = get!(firstrow, Prehashed(hashes[k]), k)
            end
            uniquerows = collect(values(firstrow))

            # Check for collisions
            collided = falses(axes(A, dim))
            @inbounds begin
                @nloops $N i A d->(if d == dim
                    k = i_d
                    j_d = uniquerow[k]
                else
                    j_d = i_d
                end) begin
                    if (@nref $N A j) != (@nref $N A i)
                        collided[k] = true
                    end
                end
            end

            if any(collided)
                nowcollided = similar(BitArray, axes(A, dim))
                while any(collided)
                    # Collect index of first row for each collided hash
                    empty!(firstrow)
                    for j = axes(A, dim)
                        collided[j] || continue
                        uniquerow[j] = get!(firstrow, Prehashed(hashes[j]), j)
                    end
                    for v ∈ values(firstrow)
                        push!(uniquerows, v)
                    end

                    # Check for collisions
                    fill!(nowcollided, false)
                    @nloops $N i A d->begin
                        if d == dim
                            k = i_d
                            j_d = uniquerow[k]
                            (!collided[k] || j_d == k) && continue
                        else
                            j_d = i_d
                        end
                    end begin
                        if (@nref $N A j) != (@nref $N A i)
                            nowcollided[k] = true
                        end
                    end
                    (collided, nowcollided) = (nowcollided, collided)
                end
            end

            (@nref $N A d->d == dim ? sort!(uniquerows) : (axes(A, d))), indexin(uniquerow, uniquerows)
        end
    end

    include("ising.jl")
    include("spectrum.jl")
    include("lattice.jl")
    include("factor.jl")
end # module
