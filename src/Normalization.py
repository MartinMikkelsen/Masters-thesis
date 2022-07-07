import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import trapz
from scipy.integrate import quad
from scipy.optimize import root
from scipy.special import spherical_jn
from scipy.integrate import solve_bvp
import seaborn as sns
import os
from pylab import plt, mpl


mpl.rcParams['font.family'] = 'XCharter'
custom_params = {"axes.spines.right": True, "axes.spines.top": True}
sns.set_theme(style="ticks", rc=custom_params)
sns.set_context("talk")

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
S = 15    #MeV
m = 135  #MeV
mn = 939  #MeV
mu = m*mn/(mn+m) #Reduced mass
g = (2*mu)
hbarc = 197.3 #MeV fm
M = m+mn
def f(r): #form factor
    return S/b*np.exp(-r**2/b**2)

def sys(r,u,E):
    y,v,I = u
    dy = v
    dv = g/(hbarc**2)*(-E+m)*y-4/r*v+g/(hbarc**2)*f(r)
    dI = 12*np.pi*f(r)*r**4*y
    return dy,dv,dI

def bc(ua, ub,E):
    ya,va,Ia = ua
    yb,vb,Ib = ub
    return va, vb+(g*(m+abs(E)))**0.5*yb, Ia, Ib-E

rmax = 5*b
rmin = 0.01*b
base1 = np.exp(1)
start = np.log(rmin)
stop = np.log(rmax)
r = np.logspace(start,stop,num=150*rmax,base=np.exp(1))
E = -2

u = [0*r,0*r,E*r/r[-1]]
res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)
#print(res.message,", E: ",res.p[0])

def plots():
    fig, ax = plt.subplots()
    plt.plot(res.x,res.y.T,'-',linewidth=3.5);
    plt.title("Numerical solution",size=15)
    plt.legend(r"$\phi$ $\phi'$ $I$".split(),loc=0);
    plt.xlabel("r [fm]")
    plt.show()

V = 1
intphi = 3*V*np.trapz(res.y.T[:,0]**2*r**2, res.x,dx=0.001)
N = 1/np.sqrt(V)*1/(np.sqrt(1+intphi))
alpha = 1/(137)

frontfactors = 16/18*np.pi*alpha*N**2*np.sqrt(2)
phi = res.y.T[:, 0]

Egamma = np.linspace(m,160,np.size(res.y.T[:,0]))
Eq = Egamma-m
k = (Eq+m)/(hbarc)
q = np.sqrt(2*mu*Eq)/hbarc

def Q(Eq):
    M = np.trapz(spherical_jn(0,q*r)*phi*r**4,r,dx=0.001)
    return abs(M)

N = []
for i in Eq:
    N.append(Q(i))
U = sum(np.array(N))

plt.figure(figsize=(9,5.5))
plt.title("$S=%s$ MeV, $b=%s$ fm" %(S,b), x=0.5, y=0.8)
plt.legend(r"$\frac{E}{12\pi}$ $\phi'$ $\phi$".split(),loc=0,frameon=False);
plt.xlabel(r"$E_\gamma$ [MeV]")
plt.ylabel(r"$\sigma$")
plt.tight_layout()

dsigmadomega = frontfactors*np.sqrt(Eq/m)*(Eq+m)/(hbarc)**3*mu**2*U**2
plt.plot(Egamma,4*np.pi*dsigmadomega,linewidth=3.5);
plt.show()
#save_fig("theorypgamma2")

#Now considering the exact matrix element
theta = 0

def F(s):
    Integral = np.trapz(spherical_jn(1, s*r)*phi*r**3, r, dx=0.001)
    return abs(Integral)

Eq = Egamma-m
k = Egamma/hbarc
q = np.sqrt(2*mu*Eq)/hbarc
s = q+mn/M*k

K = []
for i in s:
    K.append(F(i))
B = sum(np.array(K))

def dsigmadOmegaAngle(Egamma):
    frontfactors =  16/2*np.pi*np.sqrt(2)*alpha
    dsigmadOmega = frontfactors*np.sqrt(Eq/m)*(mu/m)**(3/2)*1/k*(q**2-(k*q)**2/k**2)*B**2
    return dsigmadOmega
