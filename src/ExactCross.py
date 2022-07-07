import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.integrate import quad
from scipy.optimize import root
from scipy.integrate import solve_bvp
from scipy.special import spherical_jn
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

b = 0.65    #fm
S = 12    #MeV
m = 135  #MeV
mn = 939  #MeV
mu = m*mn/(mn+m) #Reduced mass
M = m+mn
g = (2*mu)
hbarc = 197.3 #MeV fm
alpha = 1/137

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
r = np.logspace(start,stop,num=5000,base=np.exp(1))
E = -2

u = [0*r,0*r,E*r/r[-1]]
res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)

def radtodeg(x):
    degree=(x*180)/np.pi
    return degree

theta = np.linspace(0,np.pi,np.size(res.x))
phi = res.y.T[:, 0]

def F(s):
    Integral = np.trapz(spherical_jn(1,s*r)*phi*r**3, r, dx=0.001)
    return abs(Integral)

def dsigmadOmegaAngle(Egamma):
    Eq = Egamma-m
    k = Egamma/hbarc
    q = np.sqrt(2*mu*Eq)/hbarc
    s = q+mn/M*k

    frontfactors = 16/2*np.pi*np.sqrt(2)*np.sqrt(Eq/m)*(mu/m)**(3/2)

    dsigmadOmega = frontfactors*1/k*(q**2-(k*q*np.cos(theta))**2/k**2)*F(s)**2
    return dsigmadOmega

plt.figure(figsize=(9,5.5));
plt.plot(radtodeg(theta),dsigmadOmegaAngle(240)*1e6)
plt.plot(radtodeg(theta),dsigmadOmegaAngle(260)*1e6)
plt.plot(radtodeg(theta),dsigmadOmegaAngle(280)*1e6)
plt.plot(radtodeg(theta),dsigmadOmegaAngle(300)*1e6)
plt.xlabel(r"$\theta_{c.m}$ ");
plt.ylabel(r"$d\sigma/d\Omega $ [$\mu$b/sr ]");
#plt.legend(r"$(\gamma,\pi^+)$ $(\gamma,\pi^0)$".split(),loc=0,frameon=False);
