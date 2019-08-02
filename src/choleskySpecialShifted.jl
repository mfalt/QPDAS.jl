""" FS::CholeskySpecialShifted{T,MT}

Special Cholesky type to allow for "removal" of rows and columns from a Cholesky factorization.
Factorizes the shifted (positive definite A+ϵI) and used iterative refinement to solve system.

`F0 = cholesky(A+ϵI)`
`F = CholeskySpecialShifted(F0, A, ϵ)`
or
`F = CholeskySpecialShifted(A, shift=...)`

Will destroy F0!

"Removing" row and column generates the factorization corresponding to
`A[:,i] = 0; A[i,:] = 0; A[i,i] = 1`

`deleterowcol!(F::CholeskySpecialShifted, i)`
    Delete row and column i

`addrowcol!(F::CholeskySpecialShifted, i)`
    Reverse the deletion row and column i

Supports `ldiv!`

Requires roughly n^2 + 3n extra memory
Update of row k requires O(n^2) computations
Solving is equivalent to a normal cholesky

"""
struct CholeskySpecialShifted{T,MT} <: AbstractCholeskySpecial{T,MT}
    F::Cholesky{T,MT}
    idx::Set{Int}               # Indices of deleted rows and column
    tmp::Vector{T}              # tmp storage in deleterowcol
    tmp2::Vector{T}             # tmp storage in deleterowcol
    tmp3::Vector{T}             # tmp storage in ldiv2!
    tmp4::Vector{T}             # tmp storage in ldiv2!
    tmp5::Vector{T}             # tmp storage in ldiv2!
    M::MT
    shift::T
end

# TODO We should try to make some deafult choise here
function CholeskySpecialShifted(F::Cholesky{T,MT}, M, shift) where {T,MT}
    if F.uplo != 'U'
        error("Only implemented for U type")
    end
    n = size(F,1)
    CholeskySpecialShifted{T,MT}(F, Set{Int}(),                 # F and idx
        fill(zero(T), n), fill(zero(T), n), fill(zero(T), n),   # tmp, tmp2, tmp3
        fill(zero(T), n), fill(zero(T), n),                     # tmp4, tmp5
        M, shift)                                               # M (unshifted), shift
end

function CholeskySpecialShifted(M::AbstractMatrix{T}, shift=sqrt(sqrt(eps(T))*eps(T))) where T
    F = cholesky(M + I*shift)
    return CholeskySpecialShifted(F, M, shift)
end

# Helper function norm(a-b)
function normdiff(a,b)
    n = length(a)
    @assert n == length(b)
    s = zero(eltype(a))
    @inbounds @simd for i in eachindex(a)
        s += (a[i]-b[i])^2
    end
    return sqrt(s)
end

"""
b, projection = ldiv2!(F::CholeskySpecialShifted{T,MT}, b::AbstractVector{T}; x0=zero(T))

Solve `Mx=b` where F is Cholesky factorization if M with some rows and columns set to identity

The solution `x` is the projection onto `Mx=b` from `x0`.
`x0` can be a Vector or Number (treated as `fill(x0,n)`).

If no solution exists, finds projection of `b` onto space `Mx=0`.

`projection=false` if soulution `Mx=b` found, and
`projection=true` if not.

Overwrites and returns `b`.

"""
function ldiv2!(F::CholeskySpecialShifted{T,MT}, b::AbstractVector{T}; x0=zero(T), data=nothing) where {T,MT}
    done = false
    nleft = 2 # Number of iterations to run when done
    projection = false
    rk = F.tmp
    xk = F.tmp2
    rkold = F.tmp3
    xkold = F.tmp4
    xkoldold = F.tmp5

    rnorm = typemax(norm(zero(T)/one(T))) # Inf of right type
    xnorm = rnorm   # norm(x_k)
    drnorm = rnorm  # norm(r_k - r_(k-1))
    dxnorm = rnorm  # norm(x_k - x_(k-1))
    ddxnorm = rnorm # norm(x_k -2x_(k+1) + x_(k-2))

    # Initial value
    xk .= x0
    # Set rk (xkdiffold) to Inf
    rk .= rnorm

    for j in F.idx  # M should be M with some cols identity
        xk[j] = b[j] # This might ruin projection?
    end
    # Iterative refinement
    for i = 1:100
        for j in F.idx  # let xk be zero where M shouldn't act
            xk[j] = zero(T)
        end
        mul!(rk, F.M, xk)   # rk = M*xk
        rk .= b .- rk       # rk = b - M*xk
        for j in F.idx  # M should be M with some cols identity
            rk[j] = zero(T) # zero error here, This might ruing projection?
        end
        rnorm = norm(rk)
        drnorm = normdiff(rk, rkold)
        # Now save rk
        rkold .= rk
        #push!(data.rk, copy(rk))
        ldiv!(F.F, rk)      # rk = (M+ϵI)⁻¹(b - M*xk)
        for j in F.idx  # M should be M with some cols identity
            rk[j] = zero(T) # zero error here, This might ruing projection?
        end
        xk .+= rk           # xk .= xk + (M+ϵI)⁻¹(b - M*xk)

        #push!(data.xk, copy(xk))

        xnorm = norm(xk)
        dxnorm = norm(rk)
        # We can now use xkoldold as temp storage for ddx
        xkoldold .= rk .- xkold .+ xkoldold # ddx
        #rk .= rk .- xkold .+ xkoldold # ddx
        ddxnorm = norm(xkoldold)
        # Now save xk
        xkoldold .= xkold
        xkold .= xk

        # DEBUG && println("rnorm: $rnorm")
        # DEBUG && println("xnorm: $xnorm")
        # DEBUG && println("drnorm: $drnorm")
        # DEBUG && println("dxnorm: $(dxnorm)")
        # println("ddxnorm: $(ddxnorm)")
        # println("dxnorm/xnorm: $(dxnorm/xnorm)")
        # println("ddxnorm/xnorm: $(ddxnorm/xnorm)")

        # This should re robust
        # TODO 1e-11 for robustness?
        if rnorm < 1e-10 && drnorm < 1e-10 # Break if we found Mx=b solution

            done = true
            projection = false
            if nleft < 1
                DEBUG && println("Projection: $projection i: $i")
                break
            end
            nleft -= 1
        end
        # Maybe not as robust, so only use one case
        if ddxnorm/xnorm < 1e-7
            if dxnorm/xnorm < 1e-2 # Let the robust case catch this
                # done = true
                # projection = false
                # break
            else
                done = true
                projection = true
                if nleft < 1
                    DEBUG && println("Projection: $projection i: $i")
                    break
                end
                nleft -= 1
            end
        end
    end
    for j in F.idx  # M should be M with some cols identity
        xk[j] = b[j] # This might ruin projection?
    end

    if !done # Allow larger erros
        DEBUG && println("No convergence")
        if rnorm < 1e-10 # Assume solution
            if dxnorm > 1e-10 # We don't want this
                error("ldiv! did not converge to a solution, case 1, rnorm: $rnorm, dxnorm: $(dxnorm)")
            end
            projection = false
            DEBUG && println("Projection: $projection after")
        elseif ddxnorm/xnorm < 1e-5 && dxnorm/xnorm > 1e-3
            projection = true
            DEBUG && println("Projection: $projection after")
        else
            # Try to catch case where it doesnt really matter
            #@warn "Testing direction we are not sure about"
            #projection = true
            error("ldiv! did not converge to a solution, case 2, rnorm: $rnorm, dxnorm: $(dxnorm), dxnorm/xnorm: $(dxnorm/xnorm), ddxnorm/xnorm: $(ddxnorm/xnorm)")
        end
    end
    DEBUG && println("Projection: $projection at end test")
    if projection
        b .= rk.*F.shift
    else
        b .= xk
    end

    return b, projection
end


"""

ldiv!(F::CholeskySpecialShifted{T,MT}, b::AbstractVector{T}; x0=zero(T))

Solve Mx=b where F is Cholesky factorization if M with some rows and columns set to identity

"""
function LinearAlgebra.ldiv!(F::CholeskySpecialShifted{T,MT}, b::AbstractVector{T}; x0=zero(T)) where {T,MT}
    _, projection = ldiv2!(F, b, x0=x0)
    if projection
        @error "ldiv! did not converge to a solution"
        error(SingularException)
    end
    return b
end

Base.size(F::CholeskySpecialShifted) = size(F.F)
Base.size(F::CholeskySpecialShifted, i::Int) = size(F.F)[i]

function Base.getproperty(F::CholeskySpecialShifted{T,MT}, s::Symbol) where {T,MT}
    if s == :U
        return F.F.U
    else
        getfield(F,s)
    end
end

""" `deleterowcol!(F::CholeskySpecialShifted{T}, i)`
    Given F=U*U', update factorization corresponding to setting
    F[i, :] = 0, F[:, i] = 0
    F[i,i] = 1.
"""
function deleterowcol!(F::CholeskySpecialShifted{T,MT}, i) where {T,MT}
    if i in F.idx
        error("Can not delete row and column that is already deleted")
    end
    n = size(F.F,1)
    # Create vector for rank update
    F.tmp[1:i] .= zero(T)
    F.tmp[(i+1):n] .= F.F.U[i, (i+1):n]
    # Record that  this row/column is "removed"
    push!(F.idx, i)
    # Do rank update
    lowrankupdate!(F.F, F.tmp)
    # Zero out
    for i in F.idx
        F.F.U[1:(i-1),i] .= zero(T)
        F.F.U[i,(i+1):end] .= zero(T)
        F.F.U[i,i] = one(T)
    end
    return
end


"""  Reverse deleterowcol!
"""
function addrowcol!(F::CholeskySpecialShifted{T,MT}, i) where {T,MT}
    if !(i in F.idx)
        error("Can not add row and column that has not been deleted")
    end
    n = size(F.F,1)
    b = F.tmp
    b .= F.M[:,i]
    b[i] += F.shift # Add shift from [i,i]
    b[i:end] .= zero(T)
    for j in F.idx
        #This column should be zero in M
        b[j] = zero(T)
    end
    ldiv!(F.F.U', b) # b is now [S12; junk]
    S12 = view(b, 1:(i-1))
    # Scalar result
    S22 = sqrt((F.M[i,i] + F.shift) - S12'S12)

    # Get row from M (under assumption that some rows/cols are identity)
    M23 = view(F.tmp2, (i+1):n)
    M23 .= F.M[i,(i+1):end] # No shift here
    for j in F.idx
        if j > i
            #This column should be zero in M
            F.tmp2[j] = zero(T)
            # Equivalent to
            # M23[j-i] = zero(T)
        end
    end
    # TODO see if view on factor matrix directly is faster
    # Now set the new row to
    # S23 = (M23 - F.U[1:(i-1),(i+1):end]'S12)/S22 # Transposed
    S23 = view(F.F.U, i, (i+1):n) # This is where we write
    S13 = view(F.F.U, 1:(i-1), (i+1):n)
    mul!(S23, S13', S12)
    S23 .= (M23 .- S23)./S22

    # Set the diagonal
    F.F.U[i,i] = S22
    # And the column
    F.F.U[1:(i-1),i] .= S12

    # No longer need S12, so reuse tmp for rank update
    F.tmp[1:i] .= zero(T)
    F.tmp[(i+1):n] .= S23

    #Update the triangular part S33
    lowrankdowndate!(F.F, F.tmp)

    # Rememeber that i is added back
    pop!(F.idx, i)
    return
end
