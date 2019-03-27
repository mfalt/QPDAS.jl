using LinearAlgebra, QPDAS, Random, Test
import QPDAS: deleterowcol!, addrowcol!, LDLTSpecial

Random.seed!(3141592)
for ii = 1:1563
    m = 5
    A = randn(m,m);
    M = [A;A]*[A;A]';
    b = randn(2m);
    i = randperm(2m)[1:3] # 5 random, non overlapping indices

    F = ldlt!(Symmetric(copy(M),:L), pivot=true);
    FF = LDLTSpecial(F);
    #
    # diag(FF.F.D)
    # for j in 1:2m
    #     F.matrix[j,j] = max(0, F.matrix[j,j])
    #     if F.matrix[j,j] < 100eps()
    #         F.matrix[j,j] = 0
    #         F.matrix[j:end,j] .= 0
    #     end
    # end

    for j in i
        deleterowcol!(FF, j, true)
    end

    idx = BitArray(undef, 2m);
    idx .= true;
    for j in i[1:2]
        idx[j] = false
    end
    MS = M[idx, idx];
    bs = b[idx];

    FM = (FF.F.L*FF.F.D*FF.F.L')[invperm(FF.F.p), invperm(FF.F.p)];

    @test FM[idx,idx] ≈ M[idx,idx]

    for j in i
        addrowcol!(FF, j, true)
    end

    FM = (FF.F.L*FF.F.D*FF.F.L')[invperm(FF.F.p), invperm(FF.F.p)];

    F2 = ldlt!(Symmetric(copy(M),:L), pivot=true);
    @show ii
    @test FM ≈ M

end

M2 = copy(M)
M2[4,:] .= 0; M2[:,4] .= 0; M2[4,4] = 1
M2[2,:] .= 0; M2[:,2] .= 0; M2[2,2] = 1
Ft = ldlt!(Symmetric(copy(M2),:L), pin = F.p);


L = [1.0 0 0 0;
     0   1 0 0;
     0.7301169089166066 0 1 0;
     0.7301169089166066 0 1 1]
D = diagm(0 => [3.8707839746549375, 1.0, 3.0829157422185953, 0.0])
p = [2,4,3,1]
uplo = 'L'
FM = L;
for i = 1:4
    FM[i,i] = D[i,i]
end
F2 = LDLT.LDLTFactorization(FM, uplo, p, 0)


Random.seed!(3141592)
for ii = 1:1000
m = 7
A = randn(m,m);
M = [A;A]*[A;A]';
b = randn(2m);
i = randperm(2m)[1:4] # 5 random, non overlapping indices

F = ldlt!(Symmetric(copy(M),:L), pivot=true);
FF = LDLTSpecial(F);

diag(FF.F.D)
for j in 1:2m
    F.matrix[j,j] = max(0, F.matrix[j,j])
    if F.matrix[j,j] < 100eps()
        F.matrix[j,j] = 0
        F.matrix[j:end,j] .= 0
    end
end

for j in i
    deleterowcol!(FF, j, false)
end

idx = BitArray(undef, 2m);
idx .= true;
for j in i
    idx[j] = false
end
MS = M[idx, idx];
bs = b[idx];

FM = (FF.F.L*FF.F.D*FF.F.L')[invperm(FF.F.p), invperm(FF.F.p)];

@test FM[idx,idx] ≈ M[idx,idx]

for j in i
    addrowcol!(FF, j)
end

FM = (FF.F.L*FF.F.D*FF.F.L')[invperm(FF.F.p), invperm(FF.F.p)];

F2 = ldlt!(Symmetric(copy(M),:L), pivot=true);
@show ii
@test FM ≈ M

end
#
#
# M2 = copy(M);
# M2[i,:] .= 0;
# M2[:,i] .= 0;
# M2[i,i] = 1;
# F2 = ldlt!(Symmetric(copy(M2),:L), pivot=true);
#
# [diag(FF.F.D) diag(F2.D)]
#
# F.p
#
# addrowcol!(FF, 3)
# (FF.F.L*FF.F.D*FF.F.L')[invperm(FF.F.p), invperm(FF.F.p)]
# F.p
# maximum(abs, (FF.F.L*FF.F.D*FF.F.L')[invperm(FF.F.p), invperm(FF.F.p)] - M)
#
# M2 = copy(M)
# F2 = ldlt!(Symmetric(copy(M2),:L), pivot=false)
# FF2 = LDLTSpecial(F)

# Need full rank to solve systems

m, n = 30, 100
i = [6,10,25,9,8]
# i = [6,8,9,10,25]
# i = [25,10,9,8,6]
A = randn(m,n)
M = A*A'
b = randn(m)
# Get smaller matrix
idx = BitArray(undef, m)
idx .= true
for j in i
    idx[j] = false
end
MS = M[idx, idx]
bs = b[idx]
# Get reference answer
as = MS\bs

# Test
F = LDLTSpecial(ldlt!(Symmetric(copy(M),:L)))
for j in i
    deleterowcol!(F, j)
end

a = F\b
norm(a[idx]-as)
@test a[idx] ≈ as
@test a[.!idx] ≈ b[.!idx]

# Test reverse, in some other order
i = [10,6,8,25,9]
for j in i
    addrowcol!(F, j)
end


@test ldlt(Symmetric(M, :L))\b ≈ F.F\b
@test ldlt(Symmetric(M, :L)).L ≈ F.L

for j in randperm!(i)
    deleterowcol!(F, j)
end
for j in randperm!(i)
    addrowcol!(F, j)
end

@test ldlt(Symmetric(M, :L))\b ≈ F.F\b
@test ldlt(Symmetric(M, :L))\b ≈ F\b
@test ldlt(Symmetric(M, :L)).L ≈ F.L
