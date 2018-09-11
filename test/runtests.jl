using Test, Random, QPDAS

include("testCholeskySpecial.jl")

import OSQP
using LinearAlgebra, SparseArrays

Random.seed!(12345)

model = OSQP.Model()

# Test qp problem
me, mi, n = 10, 10, 100
# Equality
A = randn(me, n)
b = randn(me)
# Inequality
C = randn(mi, n)
d = randn(mi)
# Project from
z = randn(n)

M = [A;C]
# Ax=b
# Cx≥d
u = [b;fill(Inf, length(d))]
l = [b;d]


OSQP.setup!(model; P=SparseMatrixCSC{Float64}(I, n, n), l=l, A=sparse(M), u=u, verbose=false,
    eps_abs=eps(), eps_rel=eps(),
    eps_prim_inf=eps(), eps_dual_inf=eps())

OSQP.update!(model; q=-z)
results = OSQP.solve!(model)
x1 = results.x

PP = QPDAS.PolytopeProjection(A,b,C,d,z)
x2 = QPDAS.solve!(PP)
#x2 = QPDAS.solveQP(A,b,C,d,z)

@test A*x2 ≈ b atol=1e-12 # works up to 1e-14
@test minimum(C*x2 - d) > -1e-12 # works up to 1e-14

@test norm(x1-z) ≈ norm(x2-z) rtol=1e-11 # works up to 1e-12
@test x1 ≈ x2 rtol=1e-10  # works up to 1e-11

# New test
b = randn(me)
d = randn(mi)
z = randn(n)

OSQP.setup!(model; P=SparseMatrixCSC{Float64}(I, n, n), l=l, A=sparse(M), u=u, verbose=false,
    eps_abs=eps(), eps_rel=eps(),
    eps_prim_inf=eps(), eps_dual_inf=eps())

OSQP.update!(model; q=-z)
results = OSQP.solve!(model)
x1 = results.x

QPDAS.update!(PP, b=b, d=d, z=z)
x2 = QPDAS.solve!(PP)

@test_broken A*x2 ≈ b atol=1e-12 # works up to 1e-14
@test_broken minimum(C*x2 - d) > -1e-12 # works up to 1e-14

@test_broken norm(x1-z) ≈ norm(x2-z) rtol=1e-11 # works up to 1e-12
@test_broken x1 ≈ x2 rtol=1e-10  # works up to 1e-11
