# The test sets in this file are mostly intended to test the "raw" form of function objects.
# Therefore we use directly the inner constructors and avoid the more convenient outer
# constructors which may potentially optimize out some of the complexity.

@testset "Argument" begin
    func = compile(fulltrans)

    val = ev(func, [0.5], lineelt)
    @test val == ()

    val = ev(func, [0.5], Element(Simplex{1}(), 1; transform=(Shift([1.0]),)))
    @test val == (Shift([1.0]),)

    val = ev(func, [0.5], Element(Simplex{1}(), 1; dimcorr=(Updim{1,1}(2.4),)))
    @test val == (Updim{1,1}(2.4),)

    val = ev(func, [0.5], Element(Simplex{1}(), 1; dimcorr=(Updim{1,1}(2.4),), transform=(Shift([1.0]),)))
    @test val == (Updim{1,1}(2.4), Shift([1.0]))

    func = compile(dimtrans)

    val = ev(func, [0.5], lineelt)
    @test val == ()

    val = ev(func, [0.5], Element(Simplex{1}(), 1; transform=(Shift([1.0]),)))
    @test val == ()

    val = ev(func, [0.5], Element(Simplex{1}(), 1; dimcorr=(Updim{1,1}(2.4),)))
    @test val == (Updim{1,1}(2.4),)

    val = ev(func, [0.5], Element(Simplex{1}(), 1; dimcorr=(Updim{1,1}(2.4),), transform=(Shift([1.0]),)))
    @test val == (Updim{1,1}(2.4),)
end

@testset "ApplyTransform" begin
    func = compile(ApplyTransform(fulltrans, Point(1), 1))

    val = ev(func, [0.5], lineelt)
    @test val == [0.5]

    val = ev(func, [0.7], Element(Simplex{1}(), 1; transform=(Shift([-0.3]),)))
    @test val ≈ [0.4]
end

@testset "ApplyTransformGrad" begin
    func = compile(ApplyTransformGrad(fulltrans, Point(1), 1))
    val = ev(func, [0.5], lineelt)
    @test val == fill(1.0, (1,1))
end

@testset "Constant" begin
    func = compile(Constant(1.0))
    val = ev(func, [0.5], lineelt)
    @test val == fill(1.0, ())

    func = compile(Constant([1.0]))
    val = ev(func, [0.5], lineelt)
    @test val == [1.0]

    func = compile(Constant([1.0, 2.0, 3.0]))
    val = ev(func, [0.2], lineelt)
    @test val == [1.0, 2.0, 3.0]

    func = Sum((Constant([0.0]), Constant([0.0])))
    @test length(Functions.linearize(func)) == 2
end

@testset "Contract" begin
    Random.seed!(2018)
    lmx = rand(Float64, 3, 4)
    rmx = rand(Float64, 4, 2)

    pfunc = Contract(Constant(lmx), Constant(rmx))
    @test size(pfunc) == (3, 2)
    func = compile(pfunc)
    val = ev(func, [0.5], lineelt)
    @test val ≈ lmx * rmx
end

@testset "GetIndex" begin
    Random.seed!(201808041834)
    array = rand(Int, 4, 5, 6)

    # Indexing with ints and colons
    pfunc = getindex(Constant(array), 3, :, :)
    @test size(pfunc) == (5, 6)
    func = compile(pfunc)
    val = ev(func, [0.1], lineelt)
    @test size(val) == (5, 6)
    @test val == array[3, :, :]

    # Indexing with evaluables
    pfunc = getindex(Constant(array), Constant(1), :, Constant(4))
    @test size(pfunc) == (5,)
    func = compile(pfunc)
    val = ev(func, [0.1], lineelt)
    @test size(val) == (5,)
    @test val == array[1, :, 4]
end

@testset "Inflate" begin
    Random.seed!(201808031205)

    data = Constant([1.0, 2.0, 3.0, 4.0])
    index1 = Constant([1, 2, 3, 4])
    func = compile(Inflate(data, (4,), 1 => index1))
    val = ev(func, [0.1], lineelt)
    @test val == [1.0, 2.0, 3.0, 4.0]

    func = compile(Inflate(data, (7,), 1 => index1))
    val = ev(func, [0.1], lineelt)
    @test val == [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0]
end

@testset "InsertAxis" begin
    Random.seed!(201808041356)
    data = rand(Float64, 4, 5, 6)

    pfunc = InsertAxis(Constant(data), [1])
    @test size(pfunc) == (1, 4, 5, 6)
    val = ev(compile(pfunc), [0.5], lineelt)
    @test val == reshape(data, 1, 4, 5, 6)

    pfunc = InsertAxis(Constant(data), [1, 1])
    @test size(pfunc) == (1, 1, 4, 5, 6)
    val = ev(compile(pfunc), [0.5], lineelt)
    @test val == reshape(data, 1, 1, 4, 5, 6)

    pfunc = InsertAxis(Constant(data), [2])
    @test size(pfunc) == (4, 1, 5, 6)
    val = ev(compile(pfunc), [0.5], lineelt)
    @test val == reshape(data, 4, 1, 5, 6)

    pfunc = InsertAxis(Constant(data), [1, 2])
    @test size(pfunc) == (1, 4, 1, 5, 6)
    val = ev(compile(pfunc), [0.5], lineelt)
    @test val == reshape(data, 1, 4, 1, 5, 6)

    pfunc = InsertAxis(Constant(data), [1, 2, 4])
    @test size(pfunc) == (1, 4, 1, 5, 6, 1)
    val = ev(compile(pfunc), [0.5], lineelt)
    @test val == reshape(data, 1, 4, 1, 5, 6, 1)
end

@testset "Inv" begin
    Random.seed!(201808071535)

    data = rand(Float64, 1, 1)
    func = compile(inv(Constant(data)))
    @test ev(func, [0.5], lineelt) ≈ inv(data)

    data = rand(Float64, 2, 2)
    func = compile(inv(Constant(data)))
    @test ev(func, [0.5], lineelt) ≈ inv(data)

    data = rand(Float64, 3, 3)
    func = compile(inv(Constant(data)))
    @test ev(func, [0.5], lineelt) ≈ inv(data)

    data = rand(Float64, 4, 4)
    func = compile(inv(Constant(data)))
    @test ev(func, [0.5], lineelt) ≈ inv(data)
end

@testset "Monomials" begin
    Random.seed!(2018)
    pts = rand(Float64, 2, 2)
    func = compile(Monomials(Constant(pts), 3, 1))
    val = ev(func, [0.5], lineelt)
    @test val[1,:,:] == fill(0.0, 2, 2)
    @test val[2,:,:] == fill(1.0, 2, 2)
    @test val[3,:,:] ≈ pts
    @test val[4,:,:] ≈ pts .^ 2
    @test val[5,:,:] ≈ pts .^ 3
end

@testset "Neg" begin
    Random.seed!(201808081346)
    data = rand(Float64, 3, 4, 5)
    func = compile(-Constant(data))
    val = ev(func, [0.5], lineelt)
    @test val == -data
end

@testset "Product" begin
    Random.seed!(2018)
    lmx = rand(Float64, 5)
    rmx = rand(Float64, 5, 9)
    func = compile(Product((Constant(lmx), Constant(rmx))))
    val = ev(func, [0.5], lineelt)
    @test val ≈ lmx .* rmx
end

@testset "Sum" begin
    Random.seed!(2018)
    lmx = rand(Float64, 5)
    rmx = rand(Float64, 5, 9)
    func = compile(Sum(Constant(lmx), Constant(rmx)))
    val = ev(func, [0.5], lineelt)
    @test val ≈ lmx .+ rmx
end

@testset "Tupl" begin
    func = compile(Tupl((Constant(1), Constant(2.0))))
    val = ev(func, [0.5], lineelt)
    @test val == (fill(1, ()), fill(2.0, ()))
end

@testset "Zeros" begin
    func = compile(Zeros(Float64, 3, 4, 5))
    val = ev(func, [0.5], lineelt)
    @test val == zeros(Float64, 3, 4, 5)
end
