mutable struct QPStatus
    iters::Int
end

QPStatus() = QPStatus(-1)

function reset!(qps::QPStatus)
    qps.iters = 0
end
function next!(qps::QPStatus)
    qps.iters += 1
end


""" `pp = BoxConstrainedQP`
    Type for problem
    min_x 1/2xᵀGx + cᵀx, s.t dᵢ ≤ xᵢ ∀ n-m<i
    where `m=size(d,1), n=size(c,1)`.
    Only d .== 0 is allowed at the moment.
    Stores `d,G,c,sol,μλ` and some temporary variables
    Where `sol` is the solution after calling `solve!(qp)`
    G is factorization of the input `G` where
    G.M is the original input `G`
"""

struct BoxConstrainedQP{T, GT<:AbstractCholeskySpecial{T}, VT<:AbstractVector{T}}
    c::VT
    d::VT
    m::Int
    G::GT
    μλ::VT
    λdual::VT
    tmp1::VT # size(G,1)
    tmp2::VT # size(G,1)
    status::QPStatus
end


function BoxConstrainedQP(G::AbstractMatrix{T}, c::VT, d::VT; semidefinite=true, ϵ = sqrt(eps(T))) where {T, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}}
    m = size(d,1)
    n = size(G,1) - m

    @assert issymmetric(G)
    @assert all(isequal(0), d) # For now we only allow positive constraints

    GF = if semidefinite
            F = cholesky(Hermitian(G + I*ϵ))
            CholeskySpecialShifted(F, G, ϵ)
        else
            F = cholesky(Hermitian(G))
            CholeskySpecial(F, G)
        end

    bQP = BoxConstrainedQP{T,typeof(GF),VT}(
        c,d,m,
        GF,
        similar(c),  # μλ
        similar(d),  # λdual
        similar(c),  # tmp1
        similar(c),  # tmp2
        QPStatus())
    return bQP
end

# Dispatch to create factor with correct shift
new_factor(M1, bQP::BoxConstrainedQP{T,GT}) where {T, GT<:CholeskySpecial} =
    cholesky(Hermitian(M1))
new_factor(M1, bQP::BoxConstrainedQP{T,GT}) where {T, GT<:CholeskySpecialShifted} =
    cholesky(Hermitian(M1 + I*bQP.G.shift))

function run_smartstart(bQP)
    n = size(bQP.G,1) - bQP.m
    M1 = copy(bQP.G.M)
    count = 0
    for i=(n+1):size(bQP.G,1)
        if bQP.c[i] > 0
            count += 1
            M1[:,i] .= 0
            M1[i,:] .= 0
        end
    end
    DEBUG && println("started with: $count active")
    empty!(bQP.G.idx)
    for i=(n+1):size(bQP.G,1)
        if bQP.c[i] > 0
            M1[i,i] = 1
            push!(bQP.G.idx, i)
        end
    end

    F = new_factor(M1, bQP)

    bQP.G.F.UL .= F.UL
    #
    # for i = 1:length(bQP.d)
    #     if bQP.c[n+i] > 0
    #         count += 1
    #         addactive!(bQP, i)
    #     end
    # end
    # println("started with: $count active")
    return
end

## Simple QP functionality
"""
Solve min 1/2 pᵀGp+gₖᵀp, s.t pᵢ=0, ∀ i ∈ Wᵢ
where gₖ = Gx+c
return optimal p and lagrange multipliers
"""
function solveEqualityQP(bQP::BoxConstrainedQP{T}, x::AbstractVector{T}) where T
    G = bQP.G.M
    μλ = bQP.μλ
    λ = bQP.λdual
    g = bQP.tmp1
    tmp = bQP.tmp2

    mul!(g, G, x)    # g = G*x
    g .+= bQP.c      # g = Gx+c
    μλ .= .-g
    # This may throw on semidefinite, is caught in findDescent
    ldiv!(bQP.G, μλ) # G*μλ = -g
    for i in bQP.G.idx # set elements in working set to zero
        μλ[i] = zero(T)
    end

    mul!(tmp, G, μλ)
    tmp .+= g # tmp = G*μλ + g
    tmpidx = sort!([bQP.G.idx...]) # Make sure λ is in predictable order
    for (i,j) in enumerate(tmpidx)
        λ[i] = tmp[j]
    end

    return μλ, view(λ, 1:length(bQP.G.idx))
end


"""
Solve min pᵀGp+gₖᵀp, s.t pᵢ=0, ∀ i ∈ Wᵢ
where gₖ = Gx+c
return optimal p and lagrange multipliers
"""
function solveEqualityOrInfiniteDescentQP(bQP::BoxConstrainedQP{T}, x::AbstractVector{T}) where T
    G = bQP.G.M
    μλ = bQP.μλ
    λ = bQP.λdual
    g = bQP.tmp1
    tmp = bQP.tmp2

    mul!(g, G, x)    # g = G*x
    g .+= bQP.c      # g = Gx+c
    μλ .= .-g

    _, projection = ldiv2!(bQP.G, μλ) # G*μλ = -g, or projection i.e. infinite descent
    for i in bQP.G.idx # set elements in working set to zero
        DEBUG && projection && @assert abs(μλ[i]) < 1e-10
        μλ[i] = zero(T)
    end

    if !projection # Fix lambda
        mul!(tmp, G, μλ)
        tmp .+= g # tmp = G*μλ + g
        tmpidx = sort!([bQP.G.idx...]) # Make sure λ is in predictable order
        for (i,j) in enumerate(tmpidx)
            λ[i] = tmp[j]
        end
    else
        DEBUG && println("g:", g)
        DEBUG && println("μλ:", μλ)
        DEBUG && println("dot(μλ, g): $(dot(μλ, g))")
        @assert dot(μλ, g) < 0 # Make sure we got descent direction
    end

    return μλ, view(λ, 1:length(bQP.G.idx)), projection
end


function findInfiniteDescent(bQP::BoxConstrainedQP{T}, x) where T
    G = bQP.G.M
    μλ = bQP.μλ
    λ = bQP.λdual
    g = bQP.tmp1
    tmp = bQP.tmp2

    mul!(g, G, x)    # g = G*x
    g .+= bQP.c      # g = Gx+c
    μλ .= zero(T)   # We want to solve G*μλ = 0, with projection from -g
    g .= .- g       # g = - g
    ldiv!(bQP.G, μλ, x0=g) # G*μλ = 0, projected from -g
    for i in bQP.G.idx # set elements in working set to zero
        DEBUG && @assert abs(μλ[i]) < 1e-10
        μλ[i] = zero(T)
    end
    DEBUG && println("g:", -g)
    DEBUG && println("μλ:", μλ)
    DEBUG && println("dot(μλ, g): $(dot(μλ, -g))")
    @assert dot(μλ, -g) < 0 # Make sure we got descent direction (g = -g)

    return μλ, view(λ, 1:length(bQP.G.idx))
end


function findblocking(bQP::BoxConstrainedQP{T}, xk, pk::AbstractVector{T}) where T
    m = bQP.m    # Number of inequality constraints
    Wk = bQP.G.idx .- m        # Indices in λ
    inotinW = setdiff(1:m, Wk) # The indices for λ not active
    μendi = length(xk)-m       # Where inequality starts in xk, pk
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

function deleteactive!(bQP::BoxConstrainedQP, i)
    active = sort!([bQP.G.idx...]) # Make sure λ is in predictable order
    addrowcol!(bQP.G, active[i])
end

function addactive!(bQP::BoxConstrainedQP, i)
    deleterowcol!(bQP.G, (size(bQP.G,1) - bQP.m)  + i) # Inequality constraints are at end of QP.G
end

function findDescent(bQP, xk)
    local pk, λi, infinitedescent
    DEBUG && println("Finding descent")
    try
        # Expect solveEqualityQP to throw if min does not exist
        pk, λi = solveEqualityQP(bQP, xk)
        DEBUG && println("Found descent")
        infinitedescent = false
    catch
        # Find infinite descnet direction
        pk, λi = findInfiniteDescent(bQP, xk)
        DEBUG && println("Found infinite descent")
        infinitedescent = true
    end

    return pk, λi, infinitedescent
end

function findDescent(bQP::BoxConstrainedQP{T,GT}, xk) where {T, GT<:CholeskySpecialShifted}
    DEBUG && println("Finding descent")
    pk, λi, infinitedescent = solveEqualityOrInfiniteDescentQP(bQP, xk)

    DEBUG && !infinitedescent && println("Found descent")
    DEBUG &&  infinitedescent && println("Found infinite descent")
    # DEBUG && println("Finding descent")
    # try
    #     # Expect solveEqualityQP to throw if min does not exist
    #     pk, λi = solveEqualityQP(bQP, xk)
    #     DEBUG && println("Found descent")
    #     infinitedescent = false
    # catch
    #     # Find infinite descnet direction
    #     pk, λi = findInfiniteDescent(bQP, xk)
    #     DEBUG && println("Found infinite descent")
    #     infinitedescent = true
    # end

    return pk, λi, infinitedescent
end
"""
Active set method for QP as in Numerical Optimization, Nocedal Wright
    min_y,λ   1/2[y;λ]ᵀG[y;λ] + cᵀ[y;λ], s.t λ≥0
    where y ∈ R^m, λ ∈ R^n
"""
function solve!(bQP::BoxConstrainedQP{T}) where T
    #xk = fill(zero(T), m+n) # Feasible point
    # TODO better initial guess?
    xk = T.(abs.(randn(size(bQP.c,1))))
    Wk = bQP.G.idx
    for i in Wk
        xk[i] = zero(T) # Make sure we start feasible with respect to initial guess
    end
    done = false
    while !done
        next!(bQP.status) # Update iteration count and similar
        #println(bQP.status.iters)
        DEBUG && println("working set: ", Wk)
        pk, λi, infinitedescent = findDescent(bQP, xk)
        #println("infinitedescent: $infinitedescent")
        #println("norm xk: $(norm(xk)) ")
        #println("norm pk: $(norm(pk)) ")
        DEBUG && println("xk: ", xk)
        DEBUG && println("pk: ", pk)
        #println("λi: ", λi)
        if !infinitedescent && norm(pk) ≤ sqrt(eps(T))
            if all(v -> v ≥ 0, λi)
                # Optimal solution
                done = true
            else
                # Remove active constraint from Wi
                (v, idx) = findmin(λi)
                deleteactive!(bQP, idx)
            end
        else # p ≠ 0
            minα, mini = findblocking(bQP, xk, pk)
            DEBUG && println("mini: $mini")
            if minα < 0
                error("Unexpected error: minα less than 0")
            end
            # If infinite descent, go all the way, otherwise max 1
            if infinitedescent
                DEBUG && println("takning infinite descent direction, α=$minα")
                if minα == typemax(T) || mini <= 0
                    error("Dual seems to be unbounded")
                end
                xk .= xk .+ minα.*pk
                addactive!(bQP, mini)
            else
                α = min(one(T), minα)
                xk .= xk .+ α.*pk
                if minα ≤ 1
                    # mini is blocking constraint
                    addactive!(bQP, mini)
                else
                    #No blocking constraint, continue
                end
            end
        end
    end
    return xk
end
