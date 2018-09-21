module QPDAS

export QuadraticProgram, solve!, update!

using LinearAlgebra, SparseArrays

# Special type that allowes for solving M\b with some rows/columns "deleted"
include("choleskySpecial.jl")
include("LDLTSpecial.jl")

""" `pp = QuadraticProgram`
    Type for problem
    min_x 1/2||x-z||, s.t Ax=b, Cx≧d
    Stores `A,b,C,d,z,G,c,sol,μλ` and some temporary variables
    Where `sol` is the solution after calling `solve!(qp)`
    G is factorization of `[A*A' -A*C'; -C*A' C*C']`
    and `c = [-A*z + b; C*z-d]`
    Dual varaibales are available as `qp.μλ`
"""
struct QuadraticProgram{T, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}, PT, PFT}
    A::MT
    C::MT
    b::VT
    d::VT
    z::VT
    P::PT
    PF::PFT
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

function QuadraticProgram(A::MT, b::VT, C::MT, d::VT, z::VT=fill(zero(T), size(A,2)), P=I) where {T, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}}
    m = size(A,1)
    n = size(C,1)
    # Build matrix a bit more efficient
    #GQ = [A*A' -A*C'; -C*A' C*C']
    #GQ = [A*P⁻¹*A' -A*P⁻¹*C'; -C*P⁻¹*A' C*P⁻¹*C']
    GQ = similar(A, m+n, m+n)
    GQ11 = view(GQ, 1:m,1:m)
    GQ12 = view(GQ, 1:m, (m+1):(m+n))
    GQ22 = view(GQ, (m+1):(m+n), (m+1):(m+n))

    # Dispatch on type of P to get efficient solve of
    # PF = factorize(P), PiAt = PF\A', PiCt = PF\C'
    PF, PiAt, PiCt = factorizesolve(P,A,C)

    mul!(GQ11,  A, PiAt)
    mul!(GQ12, A, PiCt)
    GQ12 .= .-GQ12
    mul!(GQ22, C, PiCt)
    GQ[(m+1):(m+n), 1:m] .= GQ12'

    F = cholesky(Hermitian(GQ))
    G = CholeskySpecial(F, GQ)
    c = similar(b, m+n) # Set in update!
    QP = QuadraticProgram{T,VT,MT, typeof(P), typeof(PF)}(
        A,C,b,d,z,
        P, PF,
        G,c,copy(F),
        similar(z, length(b)+length(d)),    # μλ
        similar(z, length(d)),              # λdual
        similar(z),                         # sol
        similar(z, length(b)+length(d)),    # tmp1
        similar(z, length(b)+length(d)),    # tmp2
        similar(z))                         # tmp3

    update!(QP)
    return QP
end

# General safe implementation, Sparse LDLT doesn't support PF\A'
function factorizesolve(P, A, C)
    PF = factorize(P)
    PiAt = PF\Matrix(A')
    PiCt = PF\Matrix(C')
    return PF, PiAt, PiCt
end

function factorizesolve(P::UniformScaling{Bool}, A, C)
    if P.λ != true
        error("P not positive definite")
    end
    return P, A', C'
end

function factorizesolve(P::UniformScaling, A, C)
    # Send back Adjoint instread?
    PF = P
    PiAt = copy(A')
    PiCt = copy(C')
    PiAt ./= P.λ
    PiCt ./= P.λ
    return P, PiAt, PiCt
end

function factorizesolve(P::Matrix, A, C)
    PF = factorize(P)
    # Should work for dense case
    PiAt = PF\A'
    PiCt = PF\C'
    return PF, PiAt, PiCt
end

function solve!(QP::QuadraticProgram)
    xk = activeSetQP(QP) # Calculate dual variables
    m = size(QP.A,1)
    n = size(QP.C,1)
    mul!(QP.sol, QP.C', view(xk, (m+1):(m+n)))

    mul!(QP.tmp3, QP.A', view(xk, 1:m))
    # sol = z - A'μλ[1:m] + C'μλ[(m+1):(m+n)]
    QP.sol .= QP.z .- QP.tmp3 .+ QP.sol
    # TODO Efficiency
    sol = QP.PF\QP.sol
    QP.sol .= sol
end

function update!(QP::QuadraticProgram; b=QP.b, d=QP.d, z=QP.z)
    # TODO Up
    # We only need to update c
    m = size(QP.A,1)
    n = size(QP.C,1)
    c1 = view(QP.c, 1:m)
    c2 = view(QP.c, (m+1):(m+n))

    # TODO efficiency
    Piz = QP.PF\z
    mul!(c1, QP.A, Piz)
    c1 .= b .- c1 # c1 = -A*z + b
    mul!(c2, QP.C, Piz)
    c2 .= c2 .- d # c2 = C*z - b

    QP.z .= z
    QP.b .= b
    QP.d .= d
    return
end

"""
Solve min pᵀGp+gₖᵀp, s.t pᵢ=0, ∀ i ∈ Wᵢ
where gₖ = Gx+c
return optimal p and lagrange multipliers
"""
function solveEqualityQP(QP::QuadraticProgram{T}, x::AbstractVector{T}) where T
    G = QP.G.M
    μλ = QP.μλ
    λ = QP.λdual
    g = QP.tmp1
    tmp = QP.tmp2

    mul!(g, G, x)    # g = G*x
    g .+= QP.c      # g = Gx+c
    μλ .= .-g
    ldiv!(QP.G, μλ) # G*μλ = -g
    for i in QP.G.idx # set elements in working set to zero
        μλ[i] = zero(T)
    end

    mul!(tmp, G, μλ)
    tmp .+= g # tmp = G*μλ + g
    tmpidx = sort!([QP.G.idx...]) # Make sure λ is in predictable order
    for (i,j) in enumerate(tmpidx)
        λ[i] = tmp[j]
    end

    return μλ, view(λ, 1:length(QP.G.idx))
end


function findblocking(QP::QuadraticProgram{T}, xk, pk::AbstractVector{T}) where T
    n = size(QP.C,1)           # Number of inequality constraints in original
    Wk = QP.G.idx .- n         # Indices in λ
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

function deleteactive!(QP::QuadraticProgram, i)
    active = sort!([QP.G.idx...]) # Make sure λ is in predictable order
    addrowcol!(QP.G, active[i])
end

function addactive!(QP::QuadraticProgram, i)
    deleterowcol!(QP.G, size(QP.A,1) + i) # Inequality constraints are at end of QP.G
end

"""
Active set method for QP as in Numerical Optimization, Nocedal Wright
    min_y,λ   1/2[y;λ]ᵀG[y;λ] + cᵀ[y;λ], s.t λ≥0
    where y ∈ R^m, λ ∈ R^n
"""
function activeSetQP(QP::QuadraticProgram{T}) where T
    #xk = fill(zero(T), m+n) # Feasible point
    # TODO better initial guess?
    xk = abs.(randn(size(QP.A,1) + size(QP.C,1)))
    Wk = QP.G.idx
    for i in Wk
        xk[i] = zero(T) # Make sure we start feasible with respect to initial guess
    end
    done = false
    while !done
        #println("working set: ", Wk)
        pk, λi = solveEqualityQP(QP, xk)
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
                deleteactive!(QP, idx)
            end
        else # p ≠ 0
            minα, mini = findblocking(QP, xk, pk)
            if minα < 0
                error("minα less than 0, what to do?")
            end
            α = min(one(T), minα)
            xk .= xk .+ α.*pk
            if minα ≤ 1
                # mini is blocking constraint
                addactive!(QP, mini)
            else
                #No blocking constraint, continue
            end
        end
    end
    return xk
end

end # module
