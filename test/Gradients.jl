@testset "ApplyTransform" begin
    func = compile(grad(ApplyTransform(trans, Point{2}(), 2), 2))
    val = func([0.0, 0.0], squareelt)
    @test val == Matrix(1.0I, 2, 2)
end

@testset "Point" begin
    func = compile(grad(Point{2}(), 2))
    val = func([0.0, 0.0], squareelt)
    @test val == Matrix(1.0I, 2, 2)
end

@testset "Constant" begin
    Random.seed!(201808081405)
    func = compile(grad(Constant(rand(Float64, 3, 4, 5)), 2))
    val = func([0.0, 0.0], squareelt)
    @test val == zeros(Float64, 3, 4, 5, 2)
end

@testset "Contract" begin
    Random.seed!(201808081543)

    data = rand(Float64, 3, 2)
    func = grad(Contract(Constant(data), Point{2}()), 2)
    @test size(func) == (3, 2)
    val = compile(func)([0.0, 0.0], squareelt)
    @test val == data

    data = rand(Float64, 2, 3)
    func = grad(Contract(Point{2}(), Constant(data)), 2)
    @test size(func) == (3, 2)
    val = compile(func)([0.0, 0.0], squareelt)
    @test val == data'

    pt = Point{2}()
    func = grad(Contract(Product(InsertAxis(pt, [1]), InsertAxis(pt, [2])), pt), 2)
    @test size(func) == (2, 2)
    val = compile(func)([0.3, 0.7], squareelt)
    @test val[:] ≈ [3 * 0.3^2 + 0.7^2, 2 * 0.3 * 0.7, 2 * 0.3 * 0.7, 0.3^2 + 3 * 0.7^2]
end

@testset "Inflate" begin
    index = Constant([3, 5])
    func = grad(inflate(Point{2}(), [index], (7,)), 2)
    @test size(func) == (7, 2)
    val = compile(func)([0.0, 0.0], squareelt)
    @test val == [0.0 0.0; 0.0 0.0; 1.0 0.0; 0.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0]
end

@testset "InsertAxis" begin
    func = compile(grad(InsertAxis(Point{2}(), [1, 2]), 2))
    val = func([0.0, 0.0], squareelt)
    @test size(val) == (1, 2, 1, 2)
    @test val[1,:,1,:] == Matrix(1.0I, 2, 2)
end

@testset "Inv" begin
    pt = Point{2}()
    func = grad(inv(Sum(Product(InsertAxis(pt, [1]), InsertAxis(pt, [2])), Constant(Matrix(1.0I, 2, 2)))), 2)
    @test size(func) == (2, 2, 2)
    val = compile(func)([0.2, 0.9], squareelt)
    det = 1 + 0.2^2 + 0.9^2
    ddet1 = 2 * 0.2
    ddet2 = 2 * 0.9
    @test val[1:8] ≈ [
        -ddet1 * (1 + 0.9^2),
        0.2 * 0.9 * ddet1 - 0.9 * det,
        0.2 * 0.9 * ddet1 - 0.9 * det,
        2 * 0.2 * det - (1 + 0.2^2) * ddet1,
        2 * 0.9 * det - (1 + 0.9^2) * ddet2,
        0.2 * 0.9 * ddet2 - 0.2 * det,
        0.2 * 0.9 * ddet2 - 0.2 * det,
        -ddet2 * (1 + 0.2^2),
    ] / det^2
end

@testset "Neg" begin
    func = compile(grad(-Point{2}(), 2))
    val = func([0.0, 0.0], squareelt)
    @test val == -Matrix(1.0I, 2, 2)
end

@testset "Product" begin
    Random.seed!(201808081416)
    data = rand(Float64, 3, 4, 2)
    func = Product(Constant(data), InsertAxis(Point{2}(), [1, 1]))
    @test size(func) == (3, 4, 2)
    func = grad(func, 2)
    @test size(func) == (3, 4, 2, 2)

    val = compile(func)([0.0, 0.0], squareelt)
    @test val[:,:,1,1] == data[:,:,1]
    @test val[:,:,2,2] == data[:,:,2]
    @test val[:,:,1,2] == zeros(Float64, 3, 4)
    @test val[:,:,2,1] == zeros(Float64, 3, 4)

    func = compile(grad(Product(InsertAxis(Point{2}(), [1]), InsertAxis(Point{2}(), [2])), 2))
    val = func([0.0, 0.0], squareelt)
    @test val[:] == fill(0.0, (8,))
    val = func([0.2, 0.5], squareelt)
    @test val[:] == [0.4, 0.5, 0.5, 0.0, 0.0, 0.2, 0.2, 1.0]
end

@testset "Sum" begin
    pt = Point{2}()
    func = grad(Sum(Product(InsertAxis(pt, [1]), InsertAxis(pt, [2])), Constant(Matrix(1.0I, 2, 2))), 2)
    @test size(func) == (2, 2, 2)
    func = compile(func)
    val = func([0.0, 0.0], squareelt)
    @test val[:] == fill(0.0, (8,))
    val = func([0.2, 0.5], squareelt)
    @test val[:] == [0.4, 0.5, 0.5, 0.0, 0.0, 0.2, 0.2, 1.0]
end

@testset "Zeros" begin
    func = compile(grad(Zeros(Float64, 4, 5), 2))
    val = func([0.0, 0.0], squareelt)
    @test val == zeros(Float64, 4, 5, 2)
end
