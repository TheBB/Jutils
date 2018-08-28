using Jutils.Runtime


@testset "UnsafeView" begin
    a = zeros(4, 5, 6)
    a[:] = 1:4*5*6

    z = UnsafeView(a, :, :, :)
    @test z == a
    @test size(z) == (4, 5, 6)

    z = UnsafeView(a, 1, :, :)
    @test z == a[1,:,:]
    @test size(z) == (5, 6)

    z = UnsafeView(a, :, :, 1)
    @test z == a[:,:,1]
    @test size(z) == (4, 5)

    z = UnsafeView(a, :, :, fill(2, ()))
    @test z == a[:,:,2]
    @test size(z) == (4, 5)

    z = UnsafeView(a, 1, 2, 3)
    @test z == fill(a[1,2,3], ())
    @test size(z) == ()

    @test legal_unsafe_index(1, 2, :, :, 3) == true
    @test legal_unsafe_index(1, 2, :, :, fill(3, ())) == true
    @test legal_unsafe_index(:, :, 3) == true
    @test legal_unsafe_index(1, :, :) == true
    @test legal_unsafe_index(fill(1, ()), 2, 3) == true
    @test legal_unsafe_index(:, :) == true
    @test legal_unsafe_index(:, 1, :, 1) == false
    @test legal_unsafe_index(:, 2, :) == false
    @test legal_unsafe_index([1, 2, 3]) == false
end
