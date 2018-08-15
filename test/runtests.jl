using LinearAlgebra
using Random
using SparseArrays
using Test

using Jutils.Elements
using Jutils.Functions
using Jutils.Integration
using Jutils.Mesh
using Jutils.Transforms
using Jutils.Topologies


const lineelt = Element(Simplex{1}(), 1)
const squareelt = Element(Tensor([Simplex{1}(), Simplex{1}()]), 1)

ev(func, pt, elt) = callable(func)(pt, elt)


@testset "Transforms" begin include("Transforms.jl") end

@testset "Functions" begin include("Functions.jl") end
@testset "Gradients" begin include("Gradients.jl") end

@testset "Optimization" begin include("Optimization.jl") end

@testset "Integration" begin include("Integration.jl") end
