using BoundaryValueDiffEq

const q = 9.81
L = 1.0
tspan = (0.0,pi/2)
function simplependulum!(du,u,p,t)
    θ  = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -(q/L)*sin(θ)
end

u₀_2 = [-1.6, -1.7] # the initial guess
function bc3!(residual, sol, p, t)
    residual[1] = sol(pi/4)[1] + pi/2 # use the interpolation here, since indexing will be wrong for adaptive methods
    residual[2] = sol(pi/2)[1] - pi/2
end
bvp3 = BVProblem(simplependulum!, bc3!, u₀_2, tspan)
sol3 = solve(bvp3, Shooting(Vern7()))
plot(sol3)