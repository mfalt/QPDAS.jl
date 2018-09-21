using LDLT

import LDLT: LDLTFactorization, update!
# TODO Make possible to use 'L' (LowerTriangular)
# TODO Handle permutations in F
""" FS::LDLTSpecial{T,MT}

Special LDLTFactorization type to allow for "removal" of rows and columns from a LDLTFactorization factorization.

F0 = ldlt(A)
F = LDLTSpecial(F0)

Will destroy F0!

"Removing" row and column generates the factorization corresponding to
`A[:,i] = 0; A[i,:] = 0; A[i,i] = 1`

`deleterowcol!(F::LDLTSpecial, i)`
    Delete row and column i

`addrowcol!(F::LDLTSpecial, i)`
    Reverse the deletion row and column i

Supports `ldiv!`

Requires roughly n^2 + 3n extra memory
Update of row k requires O(n^2) computations
Solving is equivalent to a normal cholesky

"""
struct LDLTSpecial{T,MT} <: Factorization{T}
    F::LDLTFactorization{T}
    idx::Set{Int}               # Indices of deleted rows and column
    tmp::Vector{T}              # tmp storage in deleterowcol
    tmp2::Vector{T}             # tmp storage in deleterowcol
    M::MT
end

function LDLTSpecial(F::LDLTFactorization{T}, M = F.L*F.D*F.L') where {T}
    if F.uplo != 'L'
        error("Only implemented for L type")
    end
    LDLTSpecial{}(F, Set{Int}(),                    # F and idx
        fill(zero(T), size(F,1)), fill(zero(T), size(F,1)), # tmp, tmp2
        M)                                                  # M
end

"""
cholsolveexclude!(F::LDLTSpecial, b)

Solve Mx=b where F is LDLTFactorization factorization if M with some rows and columns set to identity
"""
LinearAlgebra.ldiv!(F::LDLTSpecial{T,MT}, b::AbstractVector{T}) where {T,MT} =
    ldiv!(F.F, b)

Base.size(F::LDLTSpecial) = size(F.F)
Base.size(F::LDLTSpecial, i::Int) = size(F.F)[i]

function Base.getproperty(F::LDLTSpecial{T,MT}, s::Symbol) where {T,MT}
    if s == :L
        return F.F.L
    elseif s == :D
        return F.F.D
    elseif s == :P
        return F.F.P
    elseif s == :p
        return F.F.p
    else
        getfield(F,s)
    end
end

""" `deleterowcol!(F::LDLTSpecial{T}, i)`
    Given F=U*U', update factorization corresponding to setting
    F[i, :] = 0, F[:, i] = 0
    F[i,i] = 1.
"""
function deleterowcol!(F::LDLTSpecial{T,MT}, j) where {T,MT}
    i = findfirst(isequal(j), F.F.p)::Int
    if i in F.idx
        error("Can not delete row and column that is already deleted")
    end
    n = size(F.F,1)
    # Create vector for rank update
    F.tmp[1:i] .= zero(T)
    F.tmp[(i+1):n] .= F.F.L[(i+1):n,i]
    # Record that  this row/column is "removed"
    push!(F.idx, i)
    # Do rank update
    update!(F.F, F.tmp, one(T))
    # Zero out
    for i in F.idx
        F.F.L[i, 1:(i-1)] .= zero(T)
        F.F.L[(i+1):end, i] .= zero(T)
        F.F.matrix[i,i] = one(T) # Note, we cant write to diagonal in L
    end
    return
end


"""  Reverse deleterowcol!
"""
function addrowcol!(F::LDLTSpecial{T,MT}, j) where {T,MT}
    i = findfirst(isequal(j), F.F.p)::Int
    if !(i in F.idx)
        error("Can not add row and column that has not been deleted")
    end
    n = size(F.F,1)
    b = F.tmp
    b .= F.M[:,i]
    b[i:end] .= zero(T)
    for j in F.idx
        #This column should be zero in M
        b[j] = zero(T)
    end
    ldiv!(F.F.L, b) # b is now D11*[S12; junk]
    S12 = view(b, 1:(i-1))
    # TODO handle 0s in D11 here
    for j = 1:(i-1)
        S12[j] /= F.M[j,j]
    end
    # Scalar result # TODO Efficient
    L21D11 = similar(b,i-1)
    for j = 1:(j-1)
        L21D11[j] += S12[j]*F.M[j,j]
    end
    S22 = F.M[i,i] - L21D1'S12
    if abs(S22) == zero(T)
        # TODO New zero diagonal do zero L
    end

    # Get row from M (under assumption that some rows/cols are identity)
    M23 = view(F.tmp2, (i+1):n)
    M23 .= F.M[(i+1):end, i]
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
    # S23 = (M23 - F.L[(i+1):end,1:(i-1)]'S12)/S22 # Transposed
    S23 = view(F.F.L, (i+1):n, i) # This is where we write
    S13 = view(F.F.L, (i+1):n, 1:(i-1))
    mul!(S23, S13', L21D11) # Now we have S22*S23 (D22*L32)
    S23 .= (M23 .- S23)./S22

    # Set the diagonal
    F.F.L[i,i] = S22
    # And the column
    F.F.L[i,1:(i-1)] .= S12

    # No longer need S12, so reuse tmp for rank update
    F.tmp[1:i] .= zero(T)
    F.tmp[(i+1):n] .= S23

    #Update the triangular part S33
    update!(F.F, F.tmp, -S22)

    # Rememeber that i is added back
    pop!(F.idx, i)
    return
end
