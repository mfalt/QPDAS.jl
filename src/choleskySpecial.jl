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

Requires roughly 3*n^2 memory
Update of row k requires O((n-k)^2) computations
Solving is equivalent to a normal cholesky solve plus roughly 3n read/writes

"""
struct CholeskySpecial{T,MT} <: Factorization{T}
    F::Cholesky{T,MT}
    mem::UpperTriangular{T,MT}  # Memory of rows and columns that are zeroes out
    idx::Set{Int}               # Indices of deleted rows and column
    V::UpperTriangular{T,MT}    # Vectors used for lowrankup/downdate
    tmp::Vector{T}              # tmp storage in deleterowcol
    M::MT
end

function CholeskySpecial(F::Cholesky{T,MT}) where {T,MT}
    if F.uplo != 'U'
        error("Only implemented for U type")
    end
    CholeskySpecial{T,MT}(F, copy(F.U), Set{Int}(), zero(F.U),
        fill(zero(T), size(F,1)), F.U'F.U)
end

LinearAlgebra.ldiv!(F::CholeskySpecial{T,MT}, b::AbstractVector{T}) where {T,MT} =
    cholsolveexclude!(F, b)

Base.size(F::CholeskySpecial) = size(F.F)
Base.size(F::CholeskySpecial, i::Int) = size(F.F)[i]

"""
cholsolveexclude!(F,b)

Solve Mx=b where F is factorization if M with some rows and columns set to identity
"""
function cholsolveexclude!(F::CholeskySpecial{T,MT}, b::AbstractVector{T}) where {T,MT}
    idx = F.idx
    # # remeber rows and cols before zeroing
    # for i in idx
    #     F.mem[:,i] .= F.F.U[:,i]
    #     F.mem[i,:] .= F.F.U[i,:]
    # end
    # # Clear rows and cols
    # for i in idx
    #     F.F.factors[:,i] .= zero(T)
    #     F.F.factors[i,:] .= zero(T)
    #     F.F.factors[i,i] = one(T)
    # end
    # Solve system
    ldiv!(F.F,b)
    # Write back data to F
    # for i in idx
    #     F.F.U[:,i] .= F.mem[:,i]
    #     F.F.U[i,:] .= F.mem[i,:]
    # end
    return b
end

function Base.getproperty(F::CholeskySpecial{T,MT}, s::Symbol) where {T,MT}
    if s == :U
        U = copy(F.F.U)
        # Clear rows and cols
        for i in F.idx
            U[1:(i-1),i] .= zero(T)
            U[i,(i+1):end] .= zero(T)
            #U[i,i] = one(T)
        end
        return U
    else
        getfield(F,s)
    end
end
#
# """  `v, r, c = deleterowcol!(F::CholeskySpecial{T}, i)`
#     Given F=U*U', update factorization corresponding to setting
#     F[i, :] = 0, F[j, i] = 0
#     F[i,i] = 1.
# """
# function deleterowcol_old!(F::CholeskySpecial{T,MT}, i) where {T,MT}
#     if i in F.idx
#         error("Can not delete row and column that is already deleted")
#     end
#     n = size(F.F,1)
#     # Create vector for rank update
#     F.tmp[1:i] .= zero(T)
#     F.tmp[(i+1):n] .= F.F.U[i, (i+1):n]
#     # Remember this vector in F.V
#     F.V[i,(i+1):n] .= F.F.U[i, (i+1):n]
#     # Record that  this row/column is "removed"
#     push!(F.idx, i)
#     # Do rank update
#     lowrankupdate!(F.F, F.tmp)
#     return
# end


""" `deleterowcol!(F::CholeskySpecial{T}, i)`
    Given F=U*U', update factorization corresponding to setting
    F[i, :] = 0, F[j, i] = 0
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


# """  Reverse deleterowcol!
# """
# function addrowcol_old!(F::CholeskySpecial{T,MT}, i) where {T,MT}
#     if !(i in F.idx)
#         error("Can not add row and column that has not been deleted")
#     end
#     n = size(F.F,1)
#     # Get vector that we used in delete
#     F.tmp[1:i] .= zero(T)
#     F.tmp[(i+1):n] .= F.V[i, (i+1):n]
#     # Rememebr that i is added back
#     pop!(F.idx, i)
#     lowrankdowndate!(F.F, F.tmp)
#     return
# end


"""  Reverse deleterowcol!
"""
function addrowcol!(F::CholeskySpecial{T,MT}, i) where {T,MT}
    if !(i in F.idx)
        error("Can not add row and column that has not been deleted")
    end
    n = size(F.F,1)
    b = F.M[:,i]
    b[i:end] .= 0
    for j in F.idx
        #This column should be zero in M
        b[j] = zero(T)
    end
    S12 = F.U'\b
    S12 = S12[1:(i-1)]
    S22 = sqrt(F.M[i,i] - S12'S12)
    M23 = F.M[i,(i+1):end]
    for j in F.idx
        if j > i
            #This column should be zero in M
            M23[j-i] = zero(T)
        end
    end
    S23 = (M23 - F.U[1:(i-1),(i+1):end]'S12)/S22 # Transposed
    v = [fill(0.0, i);S23]
    F.F.U[i,i] = S22
    F.F.U[i,(i+1):end] .= S23
    F.F.U[1:(i-1),i] .= S12
    lowrankdowndate!(F.F, v)
    # Rememebr that i is added back
    pop!(F.idx, i)
    return
end
