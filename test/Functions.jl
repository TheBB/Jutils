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


@testset "GetItem" begin
    srand(2018)
    array = rand(Int, 4, 5, 6)

    # Indexing with ints and colons
    pfunc = GetItem(Constant(array), (3, :, :))
    @test size(pfunc) == (5, 6)
    func = generate(pfunc)
    val = func(point=[0.1], element=Element(Simplex{1}(), 1, ()))
    @test size(val) == (5, 6)
    @test val == array[3, :, :]

    # Indexing with evaluables
    pfunc = GetItem(Constant(array), (Constant(1), :, Constant(4)))
    @test size(pfunc) == (5,)
    func = generate(pfunc)
    val = func(point=[0.1], element=Element(Simplex{1}(), 1, ()))
    @test size(val) == (5,)
    @test val == array[1, :, 4]

    # Indexing with multidimensional indices
    pfunc = GetItem(Constant(array), (Constant([1 2; 3 4]), :, :))
    @test size(pfunc) == (2, 2, 5, 6)
    func = generate(pfunc)
    val = func(point=[0.1], element=Element(Simplex{1}(), 1, ()))
    @test size(val) == (2, 2, 5, 6)
    @test val == array[[1 2; 3 4], :, :]
end


@testset "Inflate" begin
    srand(201808031205)

    data = Constant([1.0, 2.0, 3.0, 4.0])
    index1 = Constant([1, 2, 3, 4])
    func = generate(Inflate(data, (index1,), (4,)))
    val = func(point=[0.1], element=Element(Simplex{1}(), 1, ()))
    @test val == [1.0, 2.0, 3.0, 4.0]

    func = generate(Inflate(data, (index1,), (7,)))
    val = func(point=[0.1], element=Element(Simplex{1}(), 1, ()))
    @test val == [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0]
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


@testset "Outer" begin
    srand(201808031341)
    a = rand(Float64, 5)
    b = rand(Float64, 3)
    func = generate(Outer(Constant(a), Constant(b)); show=true)
    val = func(point=[0.5], element=Element(Simplex{1}(), 1, ()))
    @test val ≈ a .* b'
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
