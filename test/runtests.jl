using LinearAlgebra
using Random
using Test

using Jutils.Elements
using Jutils.Functions
using Jutils.Transforms


const lineelt = Element(Simplex{1}(), 1, ())
const squareelt = Element(Tensor([Simplex{1}(), Simplex{1}()]), 1, ())


@testset "Transforms" begin include("Transforms.jl") end

@testset "Functions" begin include("Functions.jl") end
@testset "Gradients" begin include("Gradients.jl") end
