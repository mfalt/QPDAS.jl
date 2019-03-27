using Test, Random, QPDAS
using LinearAlgebra, SparseArrays

import OSQP

Random.seed!(12345)

model = OSQP.Model()

# Test qp problem
me, mi, n = 20, 20, 1000
# Equality
A = randn(me, n)
b = randn(me)
# Inequality
C = randn(mi, n)
d = randn(mi)
# Linear term
z = randn(n)

M = [A;C]
# Ax=b
# Cx≤d
u = [b;d]
l = [b;fill(-Inf, length(d))]

OSQP.setup!(model; P=SparseMatrixCSC{Float64}(I, n, n), l=l, A=sparse(M), u=u, verbose=false,
    eps_abs=eps(), eps_rel=eps(),
    eps_prim_inf=eps(), eps_dual_inf=eps())

OSQP.update!(model; q=z)
@time results = OSQP.solve!(model)
x1 = results.x
val1 = results.info.obj_val

QP = QuadraticProgram(A,b,C,d,z)
@time x2, val2 = solve!(QP)
#x2 = solveQP(A,b,C,d,z)

@test A*x2 ≈ b atol=1e-12 # works up to 1e-14
@test maximum(C*x2 - d) < 1e-12 # works up to 1e-14

@test norm(x1-z) ≈ norm(x2-z) rtol=1e-11 # works up to 1e-12
@test x1 ≈ x2 rtol=1e-10  # works up to 1e-11

@test val1 ≈ val2 rtol=1e-11

# Test again with P=2.0I

OSQP.setup!(model; P=SparseMatrixCSC{Float64}(2.0I, n, n), l=l, A=sparse(M), u=u, verbose=false,
    eps_abs=eps(), eps_rel=eps(),
    eps_prim_inf=eps(), eps_dual_inf=eps())

OSQP.update!(model; q=z)
@time results = OSQP.solve!(model)
x1 = results.x
val1 = results.info.obj_val

QP = QuadraticProgram(A,b,C,d,z,2.0I)
@time x2, val2 = solve!(QP)
#x2 = solveQP(A,b,C,d,z)

@test A*x2 ≈ b atol=1e-12 # works up to 1e-14
@test maximum(C*x2 - d) < 1e-12 # works up to 1e-14

@test norm(x1-z) ≈ norm(x2-z) rtol=1e-11 # works up to 1e-12
@test x1 ≈ x2 rtol=1e-10  # works up to 1e-11

@test val1 ≈ val2 rtol=1e-11

# Test with semidefinite=false

OSQP.setup!(model; P=SparseMatrixCSC{Float64}(I, n, n), l=l, A=sparse(M), u=u, verbose=false,
    eps_abs=eps(), eps_rel=eps(),
    eps_prim_inf=eps(), eps_dual_inf=eps())

OSQP.update!(model; q=z)
@time results = OSQP.solve!(model)
x1 = results.x
val1 = results.info.obj_val


QP = QuadraticProgram(A,b,C,d,z, semidefinite=false)
@time x2, val2 = solve!(QP)
#x2 = solveQP(A,b,C,d,z)

@test A*x2 ≈ b atol=1e-12 # works up to 1e-14
@test maximum(C*x2 - d) < 1e-12 # works up to 1e-14

@test norm(x1-z) ≈ norm(x2-z) rtol=1e-11 # works up to 1e-12
@test x1 ≈ x2 rtol=1e-10  # works up to 1e-11

@test val1 ≈ val2 rtol=1e-11

# # New test
# b = randn(me)
# d = randn(mi)
z = randn(n)

OSQP.update!(model; q=z)
@time results = OSQP.solve!(model)
x1 = results.x
val1 = results.info.obj_val

update!(QP, z=z)
@time x2, val2 = solve!(QP)

@test A*x2 ≈ b atol=1e-12 # works up to 1e-14
@test maximum(C*x2 - d) < 1e-12 # works up to 1e-14

@test norm(x1-z) ≈ norm(x2-z) rtol=1e-11 # works up to 1e-12
@test x1 ≈ x2 rtol=1e-10  # works up to 1e-11

@test val1 ≈ val2 rtol=1e-11

# Update multiple
b = randn(me)
d = randn(mi)
z = randn(n)

# Rebuild for OSQP
u = [b;d]
l = [b;fill(-Inf, length(d))]

OSQP.setup!(model; P=SparseMatrixCSC{Float64}(I, n, n), l=l, A=sparse(M), u=u, verbose=false,
    eps_abs=eps(), eps_rel=eps(),
    eps_prim_inf=eps(), eps_dual_inf=eps())

OSQP.update!(model; q=z)
@time results = OSQP.solve!(model)
x1 = results.x
val1 = results.info.obj_val

update!(QP, z=z, b=b, d=d)
@time x2, val2 = solve!(QP)

@test A*x2 ≈ b atol=1e-12 # works up to 1e-14
@test maximum(C*x2 - d) < 1e-12 # works up to 1e-14

@test norm(x1-z) ≈ norm(x2-z) rtol=1e-11 # works up to 1e-12
@test x1 ≈ x2 rtol=1e-10  # works up to 1e-11

@test val1 ≈ val2 rtol=1e-11

#
# ## Compare gurobi
#
# env = Gurobi.Env()
# model = gurobi_model(env, f = -z, H = sparse(1.0*I, length(z), length(z)),
#     Aeq = A, beq = b, A = -C, b = -d)
#
# optimize(model)

### TEST P SPARSE

me, mi, n = 20, 20, 1000
# Equality
A = randn(me, n)
b = randn(me)
# Inequality
C = randn(mi, n)
d = randn(mi)
# Linear term
z = randn(n)

M = [A;C]
# Ax=b
# Cx≤d
u = [b;d]
l = [b;fill(-Inf, length(d))]

dig = fill(2.0,n)
dig[1:2:end] .= 4.0
dig2 = 0.3*dig
dig2[1:3:end] .= 1
P = spdiagm(-1 => dig2[1:end-1], 0 => dig, 1 => dig2[1:end-1])

OSQP.setup!(model; P=P, l=l, A=sparse(M), u=u, verbose=false,
    eps_abs=eps(), eps_rel=eps(),
    eps_prim_inf=eps(), eps_dual_inf=eps())

OSQP.update!(model; q=z)
@time results = OSQP.solve!(model)
x1 = results.x
val1 = results.info.obj_val

QP = QuadraticProgram(A,b,C,d,z,P)
@time x2, val2 = solve!(QP)
#x2 = solveQP(A,b,C,d,z)

@test A*x2 ≈ b atol=1e-12 # works up to 1e-14
@test maximum(C*x2 - d) < 1e-12 # works up to 1e-14

@test norm(x1-z) ≈ norm(x2-z) rtol=1e-11 # works up to 1e-12
@test x1 ≈ x2 rtol=1e-10  # works up to 1e-11

@test val1 ≈ val2 rtol=1e-11

## TEST UPDATE WITH P
b = randn(me)
d = randn(mi)
z = randn(n)

u = [b;d]
l = [b;fill(-Inf, length(d))]
OSQP.setup!(model; P=P, l=l, A=sparse(M), u=u, verbose=false,
    eps_abs=eps(), eps_rel=eps(),
    eps_prim_inf=eps(), eps_dual_inf=eps())
OSQP.update!(model; q=z)
@time results = OSQP.solve!(model)
x1 = results.x
val1 = results.info.obj_val

update!(QP, z=z, b=b, d=d)
@time x2, val2 = solve!(QP)

@test A*x2 ≈ b atol=1e-12 # works up to 1e-14
@test maximum(C*x2 - d) < 1e-12 # works up to 1e-14

@test norm(x1-z) ≈ norm(x2-z) rtol=1e-11 # works up to 1e-12
@test x1 ≈ x2 rtol=1e-10  # works up to 1e-11

@test val1 ≈ val2 rtol=1e-11

### TEST P DENSE

PA = randn(n,n)
P = PA*PA' + I

# Needs to be sparse for OSQP
OSQP.setup!(model; P=SparseMatrixCSC(P), l=l, A=sparse(M), u=u, verbose=false,
    eps_abs=eps(), eps_rel=eps(),
    eps_prim_inf=eps(), eps_dual_inf=eps())

OSQP.update!(model; q=z)
@time results = OSQP.solve!(model)
x1 = results.x
val1 = results.info.obj_val

QP = QuadraticProgram(A,b,C,d,z,P)
@time x2, val2 = solve!(QP)
#x2 = solveQP(A,b,C,d,z)

@test A*x2 ≈ b atol=1e-12 # works up to 1e-14
@test maximum(C*x2 - d) < 1e-12 # works up to 1e-14

@test norm(x1-z) ≈ norm(x2-z) rtol=1e-11 # works up to 1e-12
@test x1 ≈ x2 rtol=1e-10  # works up to 1e-11

@test val1 ≈ val2 rtol=1e-11
