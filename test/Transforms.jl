using LinearAlgebra
using Random

using Jutils.Transforms


@testset "Shift" begin
    Random.seed!(201808071455)

    pts = rand(Float64, 4)
    shift = rand(Float64, 4)
    @test applytrans(pts, (Shift(shift),)) == pts + shift
    @test applytrans_grad(pts, (Shift(shift),)) == Matrix(1.0I, 4, 4)
end
