# QPDAS

Quadratic Programming Dual Active Set solver using iterative refinement.

[![Build Status](https://travis-ci.com/mfalt/QPDAS.jl.svg?branch=master)](https://travis-ci.com/mfalt/QPDAS.jl)
[![codecov](https://codecov.io/gh/mfalt/QPDAS.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mfalt/QPDAS.jl)


The solver is written completely in Julia, and should be able to handle types of any precision.

The algorithm is based on a paper that is submitted to the Control and Decision Conference 2019.

Solves the mixed constraint ***positive-definite*** quadratic programming problem

    min 1/2 xᵀPx + zᵀx
    s.t Ax=b,
        Cx≤d

using a dual-active set method. Since the algorithm is solving the dual, it is very efficient when the number of inequalities is small.

At the moment, it is not possible to manually warm-start the problem.

Usage:
```julia
qp = QuadraticProgram(A, b, C, d, z=zeros(..), P=I; semidefinite=true, ϵ = sqrt(eps(T)), smartstart=true)
sol, val = solve!(qp)
```

Keyword arguments:

 - `semidefinite`: Refers to the dual problem. If `true` then iterative refinement is used to solve the linear systems in the dual. Must be `true` if the constraints of the primal are not linearly independent.
 - `ϵ`: The relaxation used for iterative refinement
 - `smartstart`: if `true` then the initial active set is guessed from the linear term in the dual. If `false`, then the initial active set is empty in the dual.

### Updating
The vectors `b,d,z` can be updated without re-factorizing the problem using
```julia
update!(QP::QuadraticProgram; b=QP.b, d=QP.d, z=QP.z)
```
The next solve will use the previous solution as an initial guess.

### Dual problem
It is also possible to directly formulate and solve the dual box-constrained problem

    min 1/2 xᵀGx + cᵀx,
    s.t dᵢ ≤ xᵢ ∀ i>n-m
    where m=size(d,1), n=size(c,1).

At the moment, only `d .== 0` is supported.

```julia
boxQP = BoxConstrainedQP(G, c, d; semidefinite=true, ϵ = sqrt(eps(T)))
run_smartstart(boxQP) # Run to set initial active ste guess
sol = solve!(boxQP)
```

The vector `c` can be efficiently updated using
```julia
boxQP.c = c
```

### Examples
A MPC example in reduced form is located in `examples/mpc.jl`, and an example of projection onto a polytope at `examples/polytope_projection.jl`.
