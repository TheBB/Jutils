using Random

using Jutils.Elements
using Jutils.Functions
using Jutils.Transforms


@testset "Argument" begin
    func = generate(trans)
    val = func(point=[0.5], element=Element(Simplex{1}(), 1, ()))
    @test val == ()

    val = func(point=[0.5], element=Element(Simplex{1}(), 1, (Shift([1.0]),)))
    @test val == (Shift([1.0]),)
end


@testset "ApplyTransform" begin
    func = generate(ApplyTransform(trans, Point{1}(), 1))

    val = func(point=[0.5], element=Element(Simplex{1}(), 1, ()))
    @test val == [0.5]

    val = func(point=[0.7], element=Element(Simplex{1}(), 1, (Shift([-0.3]),)))
    @test val ≈ [0.4]
end


@testset "Constant" begin
    func = generate(Constant(1.0))
    val = func(point=[0.5], element=Element(Simplex{1}(), 1, ()))
    @test val == fill(1.0, ())

    func = generate(Constant([1.0]))
    val = func(point=[0.5], element=Element(Simplex{1}(), 1, ()))
    @test val == [1.0]

    func = generate(Constant([1.0, 2.0, 3.0]))
    val = func(point=[0.2], element=Element(Simplex{1}(), 1, ()))
    @test val == [1.0, 2.0, 3.0]
end


@testset "Matmul" begin
    srand(2018)
    lmx = rand(Float64, 3, 4)
    rmx = rand(Float64, 4, 2)
    func = generate(Matmul(Constant(lmx), Constant(rmx)))
    val = func(point=[0.5], element=Element(Simplex{1}(), 1, ()))
    @test val ≈ lmx * rmx
end


@testset "Monomials" begin
    srand(2018)
    pts = rand(Float64, 2, 2)
    func = generate(Monomials(Constant(pts), 3))
    val = func(point=[0.5], element=Element(Simplex{1}(), 1, ()))
    @test val[1,:,:] == fill(1.0, 2, 2)
    @test val[2,:,:] ≈ pts
    @test val[3,:,:] ≈ pts .^ 2
    @test val[4,:,:] ≈ pts .^ 3
end


@testset "Product" begin
    srand(2018)
    lmx = rand(Float64, 5)
    rmx = rand(Float64, 5, 9)
    func = generate(Product(Constant(lmx), Constant(rmx)))
    val = func(point=[0.5], element=Element(Simplex{1}(), 1, ()))
    @test val ≈ lmx .* rmx
end


@testset "Sum" begin
    srand(2018)
    lmx = rand(Float64, 5)
    rmx = rand(Float64, 5, 9)
    func = generate(Sum(Constant(lmx), Constant(rmx)))
    val = func(point=[0.5], element=Element(Simplex{1}(), 1, ()))
    @test val ≈ lmx .+ rmx
end
