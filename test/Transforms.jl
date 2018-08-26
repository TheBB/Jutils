using LinearAlgebra
using Random

using Jutils.Transforms


@testset "Shift" begin
    Random.seed!(201808071455)

    pts = rand(Float64, 4)
    shift = rand(Float64, 4)
    ptworkspace = rand(Float64, 4)
    ptoutput = rand(Float64, 4)
    mxworkspace = rand(Float64, 4, 4)
    mxoutput = rand(Float64, 4, 4)

    @test applytrans((Shift(shift),), pts, ptworkspace, ptoutput) == pts + shift
    @test applytrans_grad((Shift(shift),), pts, ptworkspace, ptoutput, mxworkspace, mxoutput) == Matrix(1.0I, 4, 4)
end

@testset "Updim" begin
    Random.seed!(201808151553)

    pts = rand(Float64, 4)
    new = rand()
    ptworkspace = rand(Float64, 5)
    ptoutput = rand(Float64, 5)

    @test applytrans((Updim{4,1}(new),), pts, ptworkspace, ptoutput) == [new, pts...]
    @test applytrans((Updim{4,2}(new),), pts, ptworkspace, ptoutput) == [pts[1], new, pts[2:4]...]
    @test applytrans((Updim{4,3}(new),), pts, ptworkspace, ptoutput) == [pts[1:2]..., new, pts[3:4]...]
    @test applytrans((Updim{4,4}(new),), pts, ptworkspace, ptoutput) == [pts[1:3]..., new, pts[4]]
    @test applytrans((Updim{4,5}(new),), pts, ptworkspace, ptoutput) == [pts..., new]

    ptworkspace = rand(Float64, 2)
    ptoutput = rand(Float64, 2)
    mxworkspace = rand(Float64, 2, 2)
    mxoutput = rand(Float64, 2, 2)

    pts = rand(Float64, 1)
    @test applytrans_grad((Updim{1,1}(0.0),), pts, ptworkspace, ptoutput, mxworkspace, mxoutput) == [0.0 1.0; 1.0 0.0]
    @test applytrans_grad((Updim{1,2}(0.0),), pts, ptworkspace, ptoutput, mxworkspace, mxoutput) == [1.0 0.0; 0.0 -1.0]
end
