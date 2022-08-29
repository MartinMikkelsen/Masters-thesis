import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn
import scipy.special as scipy_bessel
from scipy.special import jv
from pyhank import qdht, iqdht, HankelTransform
from scipy.integrate import solve_bvp
from scipy import integrate
from scipy.integrate import quad
from scipy import fft
from sympy import hankel_transform, inverse_hankel_transform
import seaborn as sns
import os
from pylab import plt, mpl
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

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
m = 135.57  #MeV
mn = 938.272  #MeV
mp = 938.927  #MeV
mu = m*mn/(mn+m) #Reduced mass
Mpip = m+mn
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
r = np.logspace(start,stop,num=1000,base=np.exp(1))
E = -2

u = [0*r,0*r,E*r/r[-1]]
res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)
#print(res.message,", E: ",res.p[0])

Egamma = np.linspace(145,155,1000)
Eq = Egamma-0.5*Egamma**2/(m+mp)
k = Egamma/hbarc
q = np.sqrt(2*Mpip*Eq)/(hbarc)
theta = np.linspace(0,np.pi,np.size(Egamma))
s = np.sqrt(q**2+k**2*(mp/Mpip)**2)


phi = res.y.T[0:1000,0]

func = Spline(r,abs(phi*r**2*1/np.sqrt(np.pi/(2*s*r))))
hankelfunc = Spline(r,jv(3/2,s*r)*r)
newfunc = Spline(r,abs(phi*r**2*1/np.sqrt(np.pi/(2*s*r)))*jv(3/2,s*r)*r)

N = []
for i in s:
    N.append(Spline(r,phi*r**2*1/np.sqrt(np.pi/(2*i*r))*jv(3/2,i*r)*r))
    U = np.array(N)

Q = []
for j in range(U.size):
    Q.append(quad(U[j],0,rmax)[0])

#plt.plot(s,func(Q))

##Now do the inverse transform

Qinv = Spline(s,Q*np.sqrt(2*s*r/(np.pi))*spherical_jn(1,s*r)*s)

M = []
for i in r:
    M.append(Spline(s,Q*np.sqrt(2*s*i/(np.pi))*spherical_jn(1,s*i)*s))
    Y = np.array(M)

K = []
for j in range(Y.size):
    K.append(integrate.fixed_quad(Y[j],0,max(s))[0])

plt.plot(r,K);
plt.plot(r,phi*r**2*np.sqrt(np.pi/(2*s*r)));
