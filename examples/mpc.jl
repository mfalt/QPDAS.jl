using LinearAlgebra

"""
Go from `X'Q*X + U'R*U`
s.t. x+ = Ax+Bu
to
`U'F*U + 2U'*G*x(0) + x(0)'H*x(0)`
and M,N, where X = M*U + N*x(0)
"""
function complact_form(A, B, Q, R, n)
    nu = size(B,2)
    nx = size(A,1)
    M = zeros(n*nx, n*nu)
    for i = 0:(n-1)
        AiB = A^(i)*B
        for j = 1:(n-i)
            rows = (nx*(i+j-1)+1):(nx*(i+j))
            cols = (nu*(j-1)+1):(nu*j)
            M[rows, cols] .= AiB
        end
    end
    N = vcat([A^i for i in 1:n]...)
    # Blockdiag
    Qn = vcat([[zeros(size(Q,1), size(Q,1)*i) Q zeros(size(Q,1), size(Q,1)*(n-i-1))]  for i = 0:(n-1)]...)
    Rn = vcat([[zeros(size(R,1), size(R,1)*i) R zeros(size(R,1), size(R,1)*(n-i-1))]  for i = 0:(n-1)]...)
    F = M'Qn*M + Rn
    G = M'Qn*N
    H = M'Qn*M
    return F, G, H, M, N
end
A = [0.999  -3.008  -0.113  -1.608  ;
     0      0.986   0.048   0       ;
     0      2.083   1.009   0       ;
     0      0.053   0.05    1       ]
B = [-0.080 -0.635  ;
     -0.029 -0.014  ;
     -0.868 -0.092  ;
     -0.022 -0.002  ]
Q = Matrix{Float64}(I,4,4)
R = Matrix{Float64}(I,2,2)
n = 60
F, G, H, M, N = complact_form(A,B,Q,R,n)

x0 = 0.2.*ones(4)

nx = size(A, 1)
nu = size(B, 2)

xl = -0.2*ones(nx)
xu = 0.2*ones(nx)

Xl = vcat([xl for i in 1:n]...)
Xu = vcat([xu for i in 1:n]...)
Ul = Xl - N*x0
Uu = Xu - N*x0

UC = M

using QPDAS
nU = nu*n
z = G*x0
qp = QuadraticProgram(zeros(0, nU), zeros(0), [UC;-UC], [Uu; -Ul], z, F, ϵ=1e-6)

@time sol, val = solve!(qp)

X = M*sol + N*x0

using Plots
plot(reshape(X, 4, :)')
