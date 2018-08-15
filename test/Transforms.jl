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

@testset "Updim" begin
    Random.seed!(201808151553)

    pts = rand(Float64, 4)
    new = rand()
    @test applytrans(pts, (Updim{4,1}(new),)) == [new, pts...]
    @test applytrans(pts, (Updim{4,2}(new),)) == [pts[1], new, pts[2:4]...]
    @test applytrans(pts, (Updim{4,3}(new),)) == [pts[1:2]..., new, pts[3:4]...]
    @test applytrans(pts, (Updim{4,4}(new),)) == [pts[1:3]..., new, pts[4]]
    @test applytrans(pts, (Updim{4,5}(new),)) == [pts..., new]
end
