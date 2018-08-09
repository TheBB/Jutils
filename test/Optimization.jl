@testset "GetIndex(Constant)" begin
    func = Constant([1.0 2.0 3.0; 4.0 5.0 6.0])[:, 2]
    @test typetree(func) == [:Constant]
end

@testset "GetIndex(Inflate)" begin
    func = inflate(Constant([1.0, 2.0, 3.0]), [[1, 2, 3]], (5,))[:]
    @test typetree(func) == [:Inflate, [:Constant], [:Constant]]

    func = inflate(Constant([1.0, 2.0, 3.0]), [:], (3,))[:]
    @test typetree(func) == [:Inflate, [:Constant]]

    func = inflate(Constant([1.0, 2.0, 3.0]), [:], (3,))[1]
    @test typetree(func) == [:Inflate, [:Constant]]
end
