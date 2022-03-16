using Plots
using DifferentialEquations
using Trapz
using Roots
b = 1
S = 10
m = 135
mn = 939.5

μ = m*mn/(mn+m)
g = 1/(2*μ)

function f(r)
    return S*exp(-r^2/(b^2))
end

function diff(ϕ,r,E)
        return (ϕ[1],(E-m)*ϕ[0]-2/r*ϕ[1]-f(r))
end

ϕ₀ = [0,0]
rs = 1e-5:50:50

function ϕ_fun(E)
    prob = ODEProblem(diff,ϕ₀,rs)
    sol = solve(prob, Tsit5())
    integral = 12*π*trapz(ys[:,1]*f(rs)*rs^4,rs)
    return integral-E
end

E_true = find_zero(ϕ_func,-13)

rs = range(1e-5,50,1000)
ys = solve(diff(ϕ,r,E_true),ϕ₀,rs)

ϕ_true = ys[:,1]

print("Minimum found at E= ",E_true)
