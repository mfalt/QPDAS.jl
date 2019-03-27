using QPDAS, Random, Test, LinearAlgebra
import QPDAS: deleterowcol!, addrowcol!, CholeskySpecial

Random.seed!(3141592)

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
F = CholeskySpecial(cholesky(M))
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


@test cholesky(M)\b ≈ F.F\b
@test cholesky(M).U ≈ F.U

for j in randperm!(i)
    deleterowcol!(F, j)
end
for j in randperm!(i)
    addrowcol!(F, j)
end

@test cholesky(M)\b ≈ F.F\b
@test cholesky(M)\b ≈ F\b
@test cholesky(M).U ≈ F.U

@test size(F) == size(M)
@test size(F,1) == size(M,1)
