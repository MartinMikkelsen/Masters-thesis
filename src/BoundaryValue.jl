using BoundaryValueDiffEq
const b = 1
S = 10
m_π = 139.570
mn = 938.2
μ = m_π*mn/(m_π+mn)
g = (2*μ)

function f(r)
  S*exp(-r^2/b^2)
end

function sys(r,u,E)
  y,v,I = u
  dy = v
  dv = g*(-E+m)*y-2/r*v+g*f(r)
  dI = f(r)*r^4*y
  return dy,dv,dI
end

  function bc(ua,ub,E)
    ya,va,Ia = ua
    yb,vb,ub = ub
    return va,vb+(g*(m+abs(E)))^(0.5)*yb,Ia,Ib-E
  end

r = 10 .^(range(-5,stop=0,length=20))
E = -2

u = [0*r,0*r,E*r/r[1]]

res = BVProblem(sys, bc, u, r)
