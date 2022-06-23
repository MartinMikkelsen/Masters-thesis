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
import mpmath as mp

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
S = 10    #MeV
m = 139  #MeV
mn = 939  #MeV
mu = m*mn/(mn+m) #Reduced mass
g = (2*mu)
hbarc = 197.3 #MeV fm

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

frontfactors = 16*np.pi*alpha*N**2*(mu/m)**2/(9)

Egamma = np.linspace(m,500,np.size(res.y.T[:,0]))
q = np.sqrt(2*mu*(Egamma-m))/(hbarc)

def Q(q):
    M = np.trapz(spherical_jn(0,q*r)*res.y.T[:,0]*r**4,res.x,dx=0.001)
    return abs(M)

N = []
for i in q:
    N.append(Q(i))
U = np.array(N)

x = np.linspace(1,1500,750)
F1 = lambda x: mp.coulombf(1,2,x)
mp.plot([F1], [0,2])

plt.plot(Egamma,abs(spherical_jn(0,q*r))**2)
