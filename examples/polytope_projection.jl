using LinearAlgebra, SparseArrays, Random, QPDAS
# Number of variables
n = 1000
# Number of halfspaces
mi = 50 # Inequalities
me = 50 # Equalities

Random.seed!(1)
# One point in polytope
x0 = randn(n)

# Create polytope containing x0
# Inequality
C = Matrix{Float64}(undef, mi, n)
d = randn(mi)

# Make sure x0 is in polytope by setting sign of inequality
for i = 1:mi
    v = randn(n)
    b = randn()
    if v'x0  <= b
        C[i,:] .= v
    else
        C[i,:] .= -v
    end
    d[i] = b
end

# Create equality
A = randn(me, n)
b = A*x0

###### Solve with only inequality

# Project from here
x = randn(n)

@time qp = QuadraticProgram(zeros(0,n), zeros(0), C, d, -x, I)
@time sol, val = solve!(qp)
# Test Feasibility
maximum(C*sol-d)

###### Solve with equality and inequality

# Project from here
x = randn(n)
qp = QuadraticProgram(A, b, C, d, -x, I)
solve!(qp)
# Test Feasibility
maximum(C*sol-d)
