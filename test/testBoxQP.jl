
Random.seed!(12345)

#model = OSQP.Model()
P = [1.0 -1; -1 1]
q = [1.0, 1.0]
d = [0.0, 0]
bQP = BoxConstrainedQP(P, q, d)

sol = solve!(bQP)

@test sol == [0.0, 0.0]

P = [1.0 -1; -1 1]
q = [1.0-2.0,  1.0+2]
d = [0.0, 0]
bQP = BoxConstrainedQP(P, q, d)

sol = solve!(bQP)

@test sol == [1.0, 0.0]

P = [1.0 0; 0 0]
q = [-1.0,  0]
d = [0.0, 0]
bQP = BoxConstrainedQP(P, q, d)

sol = solve!(bQP)

@test sol[1] == 1.0
@test sol[2] >= 0
