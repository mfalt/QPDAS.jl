module QPDAS

export solveQP

using LinearAlgebra

# Special type that allowes for solving M\b with some rows/columns "deleted"
include("choleskySpecial.jl")

""" `pp = PolytopeProjection`
    Type for problem
    min_x 1/2||x-z||, s.t Ax=b, Cx≧d
    Stores `A,b,C,d,z,G,c,sol,μλ` and some temporary variables
    Where `sol` is the solution after calling `solve!(qp)`
    G is factorization of `[A*A' -A*C'; -C*A' C*C']`
    and `c = [-A*z + b; C*z-d]`
    Dual varaibales are available as `qp.μλ`
"""
struct PolytopeProjection{T, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}}
    A::MT
    C::MT
    b::VT
    d::VT
    z::VT
    G::CholeskySpecial{T,MT}
    c::VT
    Fsave::Cholesky{T,MT}
    μλ::VT
    λdual::VT
    sol::VT
    tmp1::VT # size(G,1)
    tmp2::VT # size(G,1)
    tmp3::VT # length(z)
end

function PolytopeProjection(A::MT, b::VT, C::MT, d::VT, z::VT=fill(zero(T), size(A,2))) where {T, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}}
    m = size(A,1)
    n = size(C,1)
    # Build matrix a bit more efficient
    #GQ = [A*A' -A*C'; -C*A' C*C']
    GQ = similar(A, m+n, m+n)
    GQ11 = view(GQ, 1:m,1:m)
    GQ12 = view(GQ, 1:m, (m+1):(m+n))
    GQ22 = view(GQ, (m+1):(m+n), (m+1):(m+n))
    mul!(GQ11,  A, A')
    mul!(GQ12, A, C')
    GQ12 .= .-GQ12
    mul!(GQ22, C, C')
    GQ[(m+1):(m+n), 1:m] .= GQ12'

    # Build vector
    # c = [-A*z + b; C*z-d]
    c = similar(b, m+n)
    c1 = view(c, 1:m)
    c2 = view(c, (m+1):(m+n))
    mul!(c1, A, z)
    c1 .= b .- c1
    mul!(c2, C, z)
    c2 .= c2 .- d

    F = cholesky(GQ)
    G = CholeskySpecial(F, GQ)
    PolytopeProjection{T,VT,MT}(A,C,b,d,z,G,c,copy(F),
        similar(z, length(b)+length(d)),    # μλ
        similar(z, length(d)),              # λdual
        similar(z),                         # sol
        similar(z, length(b)+length(d)),    # tmp1
        similar(z, length(b)+length(d)),    # tmp2
        similar(z))                         # tmp3
end

function solve!(PP::PolytopeProjection)
    xk = activeSetQP(PP) # Calculate dual variables
    m = size(PP.A,1)
    n = size(PP.C,1)
    mul!(PP.sol, PP.C', view(xk, (m+1):(m+n)))

    mul!(PP.tmp3, PP.A', view(xk, 1:m))
    # sol = z - A'μλ[1:m] + C'μλ[(m+1):(m+n)]
    PP.sol .= PP.z .- PP.tmp3 .+ PP.sol
end

function update!(PP::PolytopeProjection; b=PP.b, d=PP.d, z=PP.z)
    # We only need to update c
    m = size(PP.A,1)
    n = size(PP.C,1)
    c1 = view(PP.c, 1:m)
    c2 = view(PP.c, (m+1):(m+n))

    mul!(c1, PP.A, z)
    c1 .= b .- c1 # c1 = -A*z + b
    mul!(c2, PP.C, z)
    c2 .= c2 .- d # c2 = C*z - b

    PP.z .= z
    PP.b .= b
    PP.d .= d
    return
end

"""
Solve min pᵀGp+gₖᵀp, s.t pᵢ=0, ∀ i ∈ Wᵢ
where gₖ = Gx+c
return optimal p and lagrange multipliers
"""
function solveEqualityQP(PP::PolytopeProjection{T}, x::AbstractVector{T}) where T
    G = PP.G.M
    μλ = PP.μλ
    λ = PP.λdual
    g = PP.tmp1
    tmp = PP.tmp2

    mul!(g, G, x)    # g = G*x
    g .+= PP.c      # g = Gx+c
    μλ .= .-g
    ldiv!(PP.G, μλ) # G*μλ = -g
    for i in PP.G.idx # set elements in working set to zero
        μλ[i] = zero(T)
    end

    mul!(tmp, G, μλ)
    tmp .+= g # tmp = G*μλ + g
    tmpidx = sort!([PP.G.idx...]) # Make sure λ is in predictable order
    for (i,j) in enumerate(tmpidx)
        λ[i] = tmp[j]
    end

    return μλ, view(λ, 1:length(PP.G.idx))
end


function findblocking(PP::PolytopeProjection{T}, xk, pk::AbstractVector{T}) where T
    n = size(PP.C,1)           # Number of inequality constraints in original
    Wk = PP.G.idx .- n         # Indices in λ
    inotinW = setdiff(1:n, Wk) # The indices for λ not active
    μendi = length(xk)-n       # Where inequality starts in xk, pk
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

function deleteactive!(PP::PolytopeProjection, i)
    active = sort!([PP.G.idx...]) # Make sure λ is in predictable order
    addrowcol!(PP.G, active[i])
end

function addactive!(PP::PolytopeProjection, i)
    deleterowcol!(PP.G, size(PP.A,1) + i) # Inequality constraints are at end of PP.G
end

"""
Active set method for QP as in Numerical Optimization, Nocedal Wright
    min_y,λ   1/2[y;λ]ᵀG[y;λ] + cᵀ[y;λ], s.t λ≥0
    where y ∈ R^m, λ ∈ R^n
"""
function activeSetQP(PP::PolytopeProjection{T}) where T
    #xk = fill(zero(T), m+n) # Feasible point
    # TODO better initial guess?
    xk = abs.(randn(size(PP.A,1) + size(PP.C,1)))
    Wk = PP.G.idx
    for i in Wk
        xk[i] = zero(T) # Make sure we start feasible with respect to initial guess
    end
    done = false
    while !done
        #println("working set: ", Wk)
        pk, λi = solveEqualityQP(PP, xk)
        #println("xk: ", xk)
        #println("pk: ", pk)
        #println("λi: ", λi)
        if norm(pk) ≤ sqrt(eps(T))
            if all(v -> v ≥ 0, λi)
                # Optimal solution
                done = true
            else
                # Remove active constraint from Wi
                (v, idx) = findmin(λi)
                deleteactive!(PP, idx)
            end
        else # p ≠ 0
            minα, mini = findblocking(PP, xk, pk)
            if minα < 0
                error("minα less than 0, what to do?")
            end
            α = min(one(T), minα)
            xk .= xk .+ α.*pk
            if minα ≤ 1
                # mini is blocking constraint
                addactive!(PP, mini)
            else
                #No blocking constraint, continue
            end
        end
    end
    return xk
end

end # module
