using DifferentialEquations, Plots
function lorenz(du,u,p,t)
 du[1] = p[1]*(u[2]-u[1])
 du[2] = u[1]*(p[2]-u[3]) - u[2]
 du[3] = u[1]*u[2] - p[3]*u[3]
end

u0 = [1., 5., 10.]
tspan = (0., 100.)
p = (10.0,28.0,8/3)
prob = ODEProblem(lorenz, u0, tspan,p)
sol = solve(prob)
xyzt = plot(sol, plotdensity=10000,lw=1.5)
xy = plot(sol, plotdensity=10000, vars=(1,2))
xz = plot(sol, plotdensity=10000, vars=(1,3))
yz = plot(sol, plotdensity=10000, vars=(2,3))
xyz = plot(sol, plotdensity=10000, vars=(1,2,3))
plot(plot(xyzt,xyz),plot(xy, xz, yz, layout=(1,3),w=1), layout=(2,1))