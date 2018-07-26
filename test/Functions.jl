using Jutils.Functions
using Jutils.Elements


@testset "Constant" begin
    func = generate(Constant(1.0))
    val = func(points=hcat(0.5), element=Element(Simplex{1}(), 1))
    @test val == [1.0]

    func = generate(Constant([1.0]))
    val = func(points=hcat(0.5), element=Element(Simplex{1}(), 1))
    @test val == hcat(1.0)

    func = generate(Constant([1.0, 2.0, 3.0]))
    val = func(points=hcat(0.2), element=Element(Simplex{1}(), 1))
    @test val == cat([1.0, 2.0, 3.0], dims=2)
end
