
""" `pp = QuadraticProgram(A, b, C, d, z=zeros(..), P=I; kwargs...)`
    Type for problem
    `min_x 1/2xᵀPx + zᵀx, s.t Ax=b, Cx≤d`
    where `P` is positive definite.

    Keyword arguments:
    `semidefinite=true` Indicates that the dual might be semidefinite, which enables iterative refinement
    `ϵ = sqrt(eps(T))`  Relaxation used in iterative refinement (H+ϵI)
    `smartstart=true`   Enables a smart guess on initial active set
    `scaling=true`      Scales the constraints to have unit row norm

    Stores: `A,b,C,d,z,P,PF,sol,boxQP` and some temporary variables.
    `sol` is the solution after calling `solve!(qp)`,
    `PF` is a factorization of `P`,
    `boxQP::BoxConstrainedQP` representes the dual problem to be solved, with quadratic cost
    `G=[A*P⁻¹*A' A*P⁻¹*C'; C*P⁻¹*A' C*P⁻¹*C']`,
    linear cost `c = [A*z + b; C*z+d]`,
    and constraints `0≤xᵢ ∀i>size(A,1)`.
"""
struct QuadraticProgram{T, GT<:AbstractCholeskySpecial{T}, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}, PT, PFT}
    A::MT
    C::MT
    b::VT
    d::VT
    z::VT
    P::PT
    PF::PFT
    tmp3::VT
    sol::VT
    scaling::Bool
    scaleA::VT
    scaleC::VT
    boxQP::BoxConstrainedQP{T,GT,VT}
end

function QuadraticProgram(A::MT, b::VT, C::MT, d::VT, z::VT=fill(zero(T), size(A,2)), P=I; semidefinite=true, ϵ = sqrt(eps(T)), smartstart=true, scaling=true) where {T, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}}
    m = size(A,1)
    n = size(C,1)
    # Build matrix a bit more efficient
    #GQ = [A*P⁻¹*A' A*P⁻¹*C'; C*P⁻¹*A' C*P⁻¹*C']
    A = copy(A)
    C = copy(C)
    b = copy(b)
    d = copy(d)
    if scaling
        scaleA = vec(sqrt.(sum(abs2, A, dims=2)))
        scaleC = vec(sqrt.(sum(abs2, C, dims=2)))
        A .= A ./ scaleA
        C .= C ./ scaleC
        b .= b ./ scaleA
        d .= d ./ scaleC
    else
        # Empty vectors, make sure not to use them
        scaleA = VT(undef,0)
        scaleC = VT(undef,0)
    end

    GQ = similar(A, m+n, m+n)
    GQ11 = view(GQ, 1:m,1:m)
    GQ12 = view(GQ, 1:m, (m+1):(m+n))
    GQ22 = view(GQ, (m+1):(m+n), (m+1):(m+n))

    # Dispatch on type of P to get efficient solve of
    # PF = factorize(P), PiAt = PF\A', PiCt = PF\C'
    PF, PiAt, PiCt = factorizesolve(P,A,C)

    mul!(GQ11,  A, PiAt)
    mul!(GQ12, A, PiCt)

    mul!(GQ22, C, PiCt)
    GQ[(m+1):(m+n), 1:m] .= GQ12'

    # Dual linear cost
    dualc = similar(b, m+n)     # Set in update!
    duald = fill(zero(T), n)       # Always zero

    # Dual problem is simple Semidefinite Quadratic, with 0≤xᵢ ∀i>m
    boxQP = BoxConstrainedQP(Hermitian(GQ), dualc, duald; semidefinite=semidefinite, ϵ = ϵ)

    sol = similar(z)
    tmp3 = similar(z)
    QP = QuadraticProgram{T,typeof(boxQP.G), VT, MT, typeof(P), typeof(PF)}(
        A, C, b, d, z,
        P, PF,
        tmp3, sol, scaling, scaleA, scaleC, boxQP)

    # TODO smartstart already at first factorization
    # Set dual linear cost
    update!(QP)
    if smartstart
        run_smartstart(boxQP)
    end
    return QP
end

# Get status from boxQP
function Base.getproperty(QP::QuadraticProgram, s::Symbol)
    if s == :status
        return getproperty(QP.boxQP, s)
    end

    return getfield(QP, s)
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
    PiAt = Matrix(A')
    PiCt = Matrix(C')
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
    reset!(QP.status)
    # Solve dual problem
    xk = solve!(QP.boxQP)

    m = size(QP.A,1)
    n = size(QP.C,1)
    mul!(QP.sol, QP.C', view(xk, (m+1):(m+n)))

    mul!(QP.tmp3, QP.A', view(xk, 1:m))
    # sol = -P\(z + A'μλ[1:m] + C'μλ[(m+1):(m+n)])
    QP.sol .= .- QP.z .- QP.tmp3 .- QP.sol
    # TODO Efficiency
    sol = QP.PF\QP.sol
    QP.sol .= sol

    #1/2xᵀPx - zᵀx
    # TODO Efficiency
    tmp = QP.P*sol
    return sol, dot(tmp,sol)/2 + dot(sol, QP.z)
end

function update!(QP::QuadraticProgram; b=nothing, d=nothing, z=QP.z)
    # TODO ?

    # If updating b or z, do scaling before saving
    if  b !== nothing
        if QP.scaling
            QP.b .= b ./ QP.scaleA
        else
            QP.b .= b
        end
    end
    if d !== nothing
        if QP.scaling
            QP.d .= d ./ QP.scaleC
        else
            QP.d .= d
        end
    end

    # We need to update c
    m = size(QP.A,1)
    n = size(QP.C,1)
    c1 = view(QP.boxQP.c, 1:m)
    c2 = view(QP.boxQP.c, (m+1):(m+n))

    # TODO efficiency
    Piz = QP.PF\z
    mul!(c1, QP.A, Piz)
    c1 .= c1 .+ QP.b # c1 = A*z + b
    mul!(c2, QP.C, Piz)
    c2 .= c2 .+ QP.d # c2 = C*z + d

    QP.z .= z

    return
end
