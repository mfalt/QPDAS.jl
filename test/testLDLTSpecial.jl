using LinearAlgebra, QPDAS, Random
import QPDAS: deleterowcol!, addrowcol!, LDLTSpecial

Random.seed!(3141592)

A = randn(10,10)
M = A*A'
b = randn(10)

F = ldlt!(Symmetric(copy(M),:L))

FF = LDLTSpecial(F)

deleterowcol!(FF, 3)

addrowcol!(FF, 3)
M2 = copy(M)
M2[:,3] .= 0
M2[3,:] .= 0
M2[3,3] = 1
F2 = ldlt(Symmetric(M2, :L))

FF.F.L
F2.L
(FF.F.L*FF.F.D*FF.F.L')[invperm(FF.F.p), invperm(FF.F.p)] - M


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


A = randn(10,10);
A = A*A';

F = ldlt(A)
