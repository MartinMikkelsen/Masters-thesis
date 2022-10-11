using BoundaryValueDiffEq, Plots

b = 3.9    #fm
S = 45.5    #MeV
m = 135.57  #MeV
mn = 939.272  #MeV
mu = m*mn/(mn+m) #Reduced mass
M = m+mn
g = (2*mu)
hbarc = 197.3 #MeV fm

function f(r)
    return S/b*exp(-r^2/b^2)
end
xs = range(0, 10, length = 100)

plot(xs,f)

function sys!(du,u,p,t)
    ϕ = u[1]
    dϕ = u[2]
    du[1] = dϕ
    du[2] = g/(hbarc^2)*(-p+m)*ϕ-4/t*dϕ+g/(hbarc^2)*f(t)
    dI = 12*pi*f(t)*t^4*dϕ
end

function bc1!(residual, u, p, t)
    residual[1] = u[1][1] + 0.0
    residual[2] = u[end][1] - 0.0
end
tspan = (0.05,20)
bvp1 = BVProblem(sys!, bc1!, [0.0,0.0], tspan)
sol1 = solve(bvp1, GeneralMIRK4(), dt=0.1)
plot(sol1)