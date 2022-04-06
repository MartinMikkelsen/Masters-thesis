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

Integral = 12*np.pi*np.trapz(res.y.T[:,0]**2*res.x**4)

psi0 = np.sqrt(3/(4*np.pi*(1+Integral)))

print("The normalization constant, phi_0 =",psi0)
