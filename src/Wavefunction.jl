using LinearAlgebra
using DifferentialEquations

# Define constants
S = 10
b = 1
m = 139.57  # MeV
mn = 939.565378  # MeV
mu = m*mn/(mn+m) # Reduced mass
M = m+mn
g = 2*mu
hbarc = 197.3 # MeV fm
charge2 = hbarc/137

# Define a function to calculate S/b * exp(-r^2/b^2)
function f(r)
    return S/b*exp(-r^2/b^2)
end

# Define a function to calculate the system of differential equations
function sys(du, u, p, r)
    y, v, I = u
    E = p[1]
    du[1] = v
    du[2] = g/(hbarc^2)*(-E+m)*y - 4/r*v + g/(hbarc^2)*f(r) + charge2/r*y
    du[3] = 12*π*f(r)*r^4*y
end

# Define a function to calculate the boundary conditions
function bc(residual, ua, ub, p)
    ya, va, Ia = ua
    yb, vb, Ib = ub
    E = p[1]
    residual[1] = va
    residual[2] = vb + abs(g*(m + abs(E)))^0.5*yb
    residual[3] = Ia
    residual[4] = Ib - E
    
end

# Set the maximum and minimum values of r
rmax = 5*b
rmin = 0.01*b

# Create a logarithmically spaced array of r values
# Create a logarithmically spaced array of r values
r = collect(range(rmin, rmax, length=50))
# Set the value of E
E = -2

# Set the initial values for the system of differential equations
u0 = vcat(fill(0, length(r)), E .* r/r[end])

# Define the problem to be solved
prob = DifferentialEquations.TwoPointBVProblem(sys, bc, u0, r, [E])

# Solve the boundary value problem
res = DifferentialEquations.solve(prob, DifferentialEquations.Shooting(Tsit5()))

using Plots

plot(r, res.u[1, :], label="y")
plot!(r, res.u[2, :], label="v")
plot!(r, res.u[3, :], label="I")
