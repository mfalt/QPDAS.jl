using QPDAS, Random, Test, LinearAlgebra
import QPDAS: deleterowcol!, addrowcol!, CholeskySpecialShifted, ldiv2!


@testset "Projections and solutions" begin
@testset "Projections onto nullspace, $ϵ, $lambdamin, $closesol" for (ϵ, lambdamin, closesol) in [(1e-8, false, false), (1e-5, false, false), (1e-6, false, false), (1e-7, false, false),
                                 (1e-8, true , false), (1e-5, true , false), (1e-6, true , false), (1e-7, true , false),
                                 (1e-8, false, true ), (1e-5, false, true ), (1e-6, false, true ), (1e-7, false, true ),
                                 (1e-8, true , true ), (1e-5, true , true ), (1e-6, true , true ), (1e-7, true , true )]

Random.seed!(3141592)

m, n = 60, 40

A = randn(m,n)
M = A*A'
b = randn(m)

# Almost sinugar
F2 = svd(M, full=true)
F2.S[n+1] += 1e-1
F2.S[n+2] += 1e-3
F2.S[n+3] += 1e-5
if lambdamin
    M = F2.U*Diagonal(F2.S)*F2.U'
end

F = CholeskySpecialShifted(Symmetric(M), ϵ)

b = randn(m)

#  # Almost solution
x0 = randn(m)
# d contains nothing in range(M)
d = 1e-5*(I-M*pinv(M))*randn(m)
if closesol
    b = M*x0
    b .+= d
end

@testset "no sol" begin
    x = copy(b)
    #data1 = Data()
    xsol, projection = ldiv2!(F, x)#, data=data1)
    @test projection == true
    @test norm((I-M*pinv(M))*b - xsol) < 2/ϵ*1e-11
    @test norm(M*xsol) < 1e-8
    #println("xsol $closesol $lambdamin: $(dot(b, xsol)/norm(b)/norm(xsol))")
    #data2 = Data()
    x2sol, projection2 = ldiv2!(F, 0*x, x0=b)#, data=data2)
    @test norm((I-M*pinv(M))*b-x2sol) < 5/ϵ*1e-11 #This seems to work better
    #println("x2sol $closesol $lambdamin: $(dot(b, x2sol)/norm(b)/norm(x2sol))")
    @test norm(M*x2sol) < 1e-8
end

@testset "test sol zero residual?" begin
    b = M*x0
    x = copy(b)
    #data2 = Data()
    xsol, projection = ldiv2!(F, x)
    @test projection == false
    @test norm(M*xsol - b) < 1e-10
end

@testset "test sol ortogonal" begin
    # We should get x0 back
    b = M*x0
    x02 = x0 + pinv(M)*M*randn(m) # Add something orthogonal to Mx=0
    xsol, projection = ldiv2!(F, copy(b), x0=x02)
    @test projection == false
    @test norm(M*xsol - b) < 1e-12
    @test norm(xsol - x0) < 1e-4
end

@testset "test sol projection from zero" begin
    # We should get x0 back
    x0 = pinv(M)*M*randn(m) # Something orthogonal to Mx=0 (should be solution from 0)
    b = M*x0
    xsol, projection = ldiv2!(F, copy(b), x0=0)
    @test projection == false
    @test norm(M*xsol - b) < 1e-12
    @test norm(xsol - x0) < 1e-4
end


end

end
