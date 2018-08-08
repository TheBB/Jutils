@testset "Grad(ApplyTransform)" begin
    func = compile(grad(ApplyTransform(trans, Point{2}(), 2), 2))
    val = func([0.0, 0.0], squareelt)
    @test val == Matrix(1.0I, 2, 2)
end

@testset "Grad(Point)" begin
    func = compile(grad(Point{2}(), 2))
    val = func([0.0, 0.0], squareelt)
    @test val == Matrix(1.0I, 2, 2)
end

@testset "Grad(Constant)" begin
    Random.seed!(201808081405)
    func = compile(grad(Constant(rand(Float64, 3, 4, 5)), 2))
    val = func([0.0, 0.0], squareelt)
    @test val == zeros(Float64, 3, 4, 5, 2)
end

@testset "Grad(InsertAxis)" begin
    func = compile(grad(InsertAxis(Point{2}(), [1, 2]), 2))
    val = func([0.0, 0.0], squareelt)
    @test size(val) == (1, 2, 1, 2)
    @test val[1,:,1,:] == Matrix(1.0I, 2, 2)
end

@testset "Grad(Product)" begin
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
