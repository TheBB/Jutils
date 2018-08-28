@testset "1D-Lagrange-Mass" begin
    domain, _ = line(5)
    pfunc = outer(mkbasis(domain, Lagrange, 1))

    func = compile(pfunc)
    mass = integrate(func, domain, 2)
    @test mass ≈ diagm(
        -1 => fill(1/6, (5,)),
        0 => vcat([1/3], fill(2/3, (4,)), [1/3]),
        1 => fill(1/6, (5,)),
    )

    func = compile(pfunc; dense=false)
    mass = integrate(func, domain, 2)
    @test isa(mass, SparseMatrixCSC)
    @test nnz(mass) == 16
    @test mass ≈ sparse(
        [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6],
        [1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6],
        [2, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 2] / 6,
    )
end

@testset "1D-Lagrange-Lapl-Ref" begin
    domain, _ = line(5)

    pfunc = outer(grad(mkbasis(domain, Lagrange, 1), 1)[:,1])
    func = compile(pfunc; dense=false)
    mass = integrate(func, domain, 1)
    @test isa(mass, SparseMatrixCSC)
    @test nnz(mass) == 16
    @test mass ≈ sparse(
        [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6],
        [1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6],
        [1, -1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 1],
    )
end

@testset "1D-Lagrange-Lapl" begin
    domain, geom = line(5)

    pfunc = outer(grad(mkbasis(domain, Lagrange, 1), geom)[:,1])
    func = compile(pfunc; dense=false)
    mass = integrate(func, domain, 1)
    @test isa(mass, SparseMatrixCSC)
    @test nnz(mass) == 16
    @test mass ≈ sparse(
        [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6],
        [1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6],
        [1, -1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 1],
    )

    pfunc = outer(grad(mkbasis(domain, Lagrange, 1), 2 .* geom)[:,1])
    func = compile(pfunc; dense=false)
    mass = integrate(func, domain, 1)
    @test isa(mass, SparseMatrixCSC)
    @test nnz(mass) == 16
    @test mass ≈ sparse(
        [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6],
        [1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6],
        [1, -1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 1] / 4,
    )

    pfunc = outer(grad(mkbasis(domain, Lagrange, 1), 0.5 .* geom)[:,1])
    func = compile(pfunc; dense=false)
    mass = integrate(func, domain, 1)
    @test isa(mass, SparseMatrixCSC)
    @test nnz(mass) == 16
    @test mass ≈ sparse(
        [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6],
        [1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6],
        [1, -1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 1] * 4,
    )
end

@testset "2D-Lagrange-Mass" begin
    domain, _ = rectilinear(2, 2)
    pfunc = outer(mkbasis(domain, Lagrange, 1))

    func = compile(pfunc; dense=false)
    mass = integrate(func, domain, 3)
    @test isa(mass, SparseMatrixCSC)
    @test nnz(mass) == 49
    @test Matrix(mass) ≈ sparse(
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9],
        [1, 2, 4, 5, 1, 2, 3, 4, 5, 6, 2, 3, 5, 6, 1, 2, 4, 5, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9,
         2, 3, 5, 6, 8, 9, 4, 5, 7, 8, 4, 5, 6, 7, 8, 9, 5, 6, 8, 9],
        [4, 2, 2, 1, 2, 8, 2, 1, 4, 1, 2, 4, 1, 2, 2, 1, 8, 4, 2, 1, 1, 4, 1, 4, 16, 4, 1, 4, 1,
         1, 2, 4, 8, 1, 2, 2, 1, 4, 2, 1, 4, 1, 2, 8, 2, 1, 2, 2, 4] / 36,
    )
end

@testset "2D-Lagrange-Lapl" begin
    domain, geom = rectilinear(2, 2)
    pfunc = dropdims(sum(outer(grad(mkbasis(domain, Lagrange, 1), geom)), (3,)), (3,))

    func = compile(pfunc; dense=false)
    mass = integrate(func, domain, 3)
    @test isa(mass, SparseMatrixCSC)
    @test nnz(mass) == 49
    @test mass ≈ sparse(
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9],
        [1, 2, 4, 5, 1, 2, 3, 4, 5, 6, 2, 3, 5, 6, 1, 2, 4, 5, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9,
         2, 3, 5, 6, 8, 9, 4, 5, 7, 8, 4, 5, 6, 7, 8, 9, 5, 6, 8, 9],
        [4, -1, -1, -2, -1, 8, -1, -2, -2, -2, -1, 4, -2, -1, -1, -2, 8, -2, -1, -2, -2, -2, -2, -2, 16, -2, -2,
         -2, -2, -2, -1, -2, 8, -2, -1, -1, -2, 4, -1, -2, -2, -2, -1, 8, -1, -2, -1, -1, 4] / 6,
    )
end
