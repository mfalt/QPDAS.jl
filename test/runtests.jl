using Test, Random, QPDAS
using LinearAlgebra, SparseArrays

@testset "CholeskySpecial" begin
    include("testCholeskySpecial.jl")
end

@testset "testCholeskySpecialShifted" begin
    include("testCholeskySpecialShifted.jl")
end

@testset "testCholeskySpecialShifted" begin
    include("testCholeskySpecialShiftedProjection.jl")
end

@testset "testBoxQP" begin
    include("testBoxQP.jl")
end

@testset "testQPNoScaling" begin
    include("testQPNoScaling.jl")
end

@testset "testQP" begin
    include("testQP.jl")
end

@testset "testQPsemidefinite" begin
    include("testQPsemidefinite.jl")
end
