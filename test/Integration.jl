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

    func = compile(optimize(pfunc))
    mass = integrate(func, domain, 2)
    @test mass ≈ diagm(
        -1 => fill(1/6, (5,)),
        0 => vcat([1/3], fill(2/3, (4,)), [1/3]),
        1 => fill(1/6, (5,)),
    )

    func = compile(optimize(pfunc); dense=false)
    mass = integrate(func, domain, 2)
    @test isa(mass, SparseMatrixCSC)
    @test nnz(mass) == 16
    @test mass ≈ sparse(
        [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6],
        [1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6],
        [2, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 2] / 6,
    )
end
