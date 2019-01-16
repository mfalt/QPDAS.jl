""" FS::CholeskySpecial{T,MT}

Special Cholesky type to allow for "removal" of rows and columns from a Cholesky factorization.

F0 = cholesky(A)
F = CholeskySpecial(F0)

Will destroy F0!

"Removing" row and column generates the factorization corresponding to
`A[:,i] = 0; A[i,:] = 0; A[i,i] = 1`

`deleterowcol!(F::CholeskySpecial, i)`
    Delete row and column i

`addrowcol!(F::CholeskySpecial, i)`
    Reverse the deletion row and column i

Supports `ldiv!`

Requires roughly n^2 + 3n extra memory
Update of row k requires O(n^2) computations
Solving is equivalent to a normal cholesky

"""
struct CholeskySpecial{T,MT} <: AbstractCholeskySpecial{T,MT}
    F::Cholesky{T,MT}
    idx::Set{Int}               # Indices of deleted rows and column
    tmp::Vector{T}              # tmp storage in deleterowcol
    tmp2::Vector{T}             # tmp storage in deleterowcol
    M::MT
end

function CholeskySpecial(F::Cholesky{T,MT}, M = F.U'F.U) where {T,MT}
    if F.uplo != 'U'
        error("Only implemented for U type")
    end
    CholeskySpecial{T,MT}(F, Set{Int}(),                    # F and idx
        fill(zero(T), size(F,1)), fill(zero(T), size(F,1)), # tmp, tmp2
        M)                                                  # M
end

"""
ldiv!(F::CholeskySpecial, b)

Solve Mx=b where F is Cholesky factorization if M with some rows and columns set to identity
"""
LinearAlgebra.ldiv!(F::CholeskySpecial{T,MT}, b::AbstractVector{T}) where {T,MT} =
    ldiv!(F.F, b)

Base.size(F::CholeskySpecial) = size(F.F)
Base.size(F::CholeskySpecial, i::Int) = size(F.F)[i]

function Base.getproperty(F::CholeskySpecial{T,MT}, s::Symbol) where {T,MT}
    if s == :U
        return F.F.U
    else
        getfield(F,s)
    end
end

""" `deleterowcol!(F::CholeskySpecial{T}, i)`
    Given F=U*U', update factorization corresponding to setting
    F[i, :] = 0, F[:, i] = 0
    F[i,i] = 1.
"""
function deleterowcol!(F::CholeskySpecial{T,MT}, i) where {T,MT}
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
function addrowcol!(F::CholeskySpecial{T,MT}, i) where {T,MT}
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
    ldiv!(F.F.U', b) # b is now [S12; junk]
    S12 = view(b, 1:(i-1))
    # Scalar result
    S22 = sqrt(F.M[i,i] - S12'S12)

    # Get row from M (under assumption that some rows/cols are identity)
    M23 = view(F.tmp2, (i+1):n)
    M23 .= F.M[i,(i+1):end]
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
