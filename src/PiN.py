import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import root
from scipy.special import spherical_jn
from scipy.integrate import solve_bvp
import seaborn as sns
sns.set_style("dark")
sns.set(font_scale=1)
sns.set_style("ticks")


b = 1     #fm
S = 10    #MeV
m = 139.570   #MeV
mn = 939.5  #MeV
mu = m*mn/(mn+m) #Reduced mass
g = (2*mu)

def f(r): #form factor
    return S*np.exp(-r**2/b**2)

def sys(r,u,E):
    y,v,I = u
    dy = v
    dv = g*(-E+m)*y-2/r*v+g*f(r)
    dI = f(r)*r**4*y
    return dy,dv,dI

def bc(ua, ub,E):
    ya,va,Ia = ua
    yb,vb,Ib = ub
    return va, vb+(g*(m+abs(E)))**0.5*yb, Ia, Ib-E

r = np.logspace(-5,0,20)*5
u = [0*r,0*r,E*r/r[-1]]
E = -2
res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-5)
print(res.message,", E: ",res.p[0])

plt.plot(res.x,res.y.T,'--',linewidth=2.5);
plt.title("Numerical solution")
plt.grid(); plt.legend(r"$\phi$ $\phi'$ $I$".split());
plt.xlabel("r [fm]")
plt.savefig("Integralplot.pdf", format="pdf", bbox_inches="tight")
plt.show()
