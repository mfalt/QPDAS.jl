module QPDAS

global const DEBUG = false

export QuadraticProgram, BoxConstrainedQP, solve!, update!, ldiv2!

using LinearAlgebra, SparseArrays

abstract type AbstractCholeskySpecial{T,MT} <: Factorization{T} end

# Special type that allowes for solving M\b with some rows/columns "deleted"
include("choleskySpecial.jl")
include("choleskySpecialShifted.jl")
#include("LDLTSpecial.jl")
# The type for representing the dual (semidefinite) problem
include("boxConstrainedQP.jl")
# Teh main type for solving positive definite quadratic programs
include("quadraticProgram.jl")


end # module
