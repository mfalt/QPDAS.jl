module QPDAS

using LinearAlgebra

# Special type that allowes for solving M\b with some rows/columns "deleted"
include(choleskySpecial.jl)

""" Q, q, m, n = dual(A,b,C,d)
Given
min_x 1/2||x-z||, s.t Ax=b, Cx≧d
get the dual problem
min_y,λ   [y;λ]ᵀQ[y;λ] + qᵀ[y;λ], s.t λ≧0
where y ∈ R^m, λ ∈ R^n
"""
function dual(A,b,C,d,z)
    Q = [A*A' -A*C';
         -C*A' C*C']
    q = [-A*z + b; C*z-d]
    return Q, q, size(A,1), size(C,1)
end

"""
Solve min pᵀGp+gₖᵀp, s.t pᵢ=0, ∀ i ∈ Wᵢ
where gₖ = Gx+c
return optimal p and lagrange multipliers
"""
function solveEqualityQP(m, G::AbstractMatrix{T}, c::AbstractVector{T}, x, W) where T
    A = fill(zero(T), length(W), length(c))
    for (i,j) in enumerate(W)
        A[i,j+m] = one(T)
    end
    M = [G A'; A fill(zero(T), size(A,1), size(A,1))]
    g = G*x+c
    b = [-g; fill(zero(T), size(A,1))]
    y = M\b
    return y[1:size(G,1)], -y[(size(G,1)+1):end]
end

function findblocking(n, Wk, xk, pk::AbstractVector{T}) where T
    inotinW = setdiff(1:n, Wk) # The indices for λ
    μendi = length(xk)-n
    minα = typemax(T)
    mini = -1
    for i in inotinW
        if pk[μendi+i] < 0 # aᵢᵀpₖ < 0
            # Note: bᵢ = 0
            v = -xk[μendi+i]/pk[μendi+i]
            if  v < minα
                minα = v
                mini = i
            end
        end
    end
    return minα, mini
end

"""
Active set method for QP as in Numerical Optimization, Nocedal Wright
    min_y,λ   1/2[y;λ]ᵀG[y;λ] + cᵀ[y;λ], s.t λ≥0
    where y ∈ R^m, λ ∈ R^n
"""
function activeSetQP(G::AbstractMatrix{T}, c::AbstractVector{T}, m, n) where T
    #xk = fill(zero(T), m+n) # Feasible point
    xk = abs.(randn(m+n))
    Wk = Int[]
    done = false
    while !done
        println("working set: ", Wk)
        pk, λi = solveEqualityQP(m, G, c, xk, Wk)
        println("xk: ", xk)
        println("pk: ", pk)
        println("λi: ", λi)
        if norm(pk) ≤ sqrt(eps(T))
            if all(v -> v ≥ 0, λi)
                # Optimal solution
                done = true
            else
                # Remove active constraint from Wi
                (v, idx) = findmin(λi)
                deleteat!(Wk, idx)
            end
        else # p ≠ 0
            minα, mini = findblocking(n, Wk, xk, pk)
            if minα < 0
                error("minα less than 0, what to do?")
            end
            α = min(one(T), minα)
            xk .= xk .+ α.*pk
            if minα ≤ 1
                # mini is blocking constraint
                push!(Wk, mini)
            else
                #No blocking constraint, continue
            end
        end
    end
    return xk
end

""" x = solveQP(A,b,C,d,z)
solve
min_x ||x-z||, s.t Ax=b, Cx≧d
"""
function solveQP(A,b,C,d,z)
    Q,q,m,n = dual(A,b,C,d,z)
    μλ = activeSetQP(Q,q, m, n)
    return z0 - A'μλ[1:m] + C'μλ[(m+1):(m+n)]
end

end # module
