import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import root
from scipy.special import spherical_jn
from scipy.integrate import solve_bvp
import seaborn as sns
import os

PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".pdf", format='pdf',bbox_inches="tight")

b = 1     #fm
S = 10    #MeV
m = 139.570   #MeV
mn = 938.2  #MeV
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
E = -2

u = [0*r,0*r,E*r/r[-1]]
res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-5)
print(res.message,", E: ",res.p[0])

def plots():
    fig, ax = plt.subplots()
    plt.plot(res.x,res.y.T,'-',linewidth=2.5);
    plt.title("Numerical solution",size=15)
    plt.grid(); plt.legend(r"$\phi$ $\phi'$ $I$".split(),loc=0);
    plt.xlabel("r [fm]")
    plt.show()

V = 1
Integral = 12*np.pi*V*np.trapz(res.y.T[:,0]**2*res.x**4)

psi0 = 1*np.sqrt(V)*1/(np.sqrt(1+Integral))

print("The normalization constant, phi_0 =",psi0)
factors = psi0*2*np.sqrt(6)*np.pi**(3/2)
gamma = np.linspace(m+0.01,140,160)
q = np.sqrt(2*mu/(gamma-m))

def matrixelement(q):
    Q = factors*np.trapz(spherical_jn(0,q*res.x)*(-1)*res.y.T[:,0]*res.x**3)
    return Q

print("The matrix element = ", matrixelement(q))
fig, ax = plt.subplots()
plt.plot(res.x,(-1)*res.y.T[:,0]*res.x**3,'-',linewidth=2.5);
plt.plot(res.x,spherical_jn(0,q*res.x),linewidth=2.5)
plt.ylim(-0.25, 0.25)
plt.title("Numerical solution",size=15)
plt.grid(); plt.legend(r"$-\phi(r)$r $j_0(qr)$ $E$".split(),loc=0);
plt.xlabel("r [fm]");

plt.figure()

def matrixsquared(i):
    I = abs(factors*np.trapz(spherical_jn(0,i*res.x)*(-1)*res.y.T[:,0]*res.x**3))**2
    return I

M = []
for i in range(0,160):
    M.append(matrixsquared(q[i]))

plt.plot(q,M,'-',linewidth=2.5);
plt.xlabel('q [MeV]');
plt.grid();
plt.ylabel(r'$\sigma$ [Arb. Units]');
plt.title(r'Cross section, $E_\gamma=140$',size=15);
save_fig("Cross140")
