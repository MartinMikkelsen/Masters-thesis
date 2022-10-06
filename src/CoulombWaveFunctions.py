import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import os
import scipy as sp
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from tqdm import tqdm
import mpmath
from pylab import plt, mpl
import seaborn as sns
from numpy import frompyfunc
from scipy.special import gamma, factorial
from scipy.special import spherical_jn
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

m = 139.57039  #MeV
mp = 939.565378  #MeV
mu = m*mp/(mp+m) #Reduced mass
g = 2*mu
hbarc = 197.327 #MeV fm
alpha = 1/137
charge2 = hbarc/(137)
Mpip = m+mp

def diffcross(Egamma,S,b,theta):

    Eq = Egamma-m-0.5*Egamma**2/(Mpip)
    if Eq.any()<0 : return 0
    k = Egamma/hbarc
    q = np.sqrt(2*mu*Eq)/(hbarc)
    s = np.sqrt(q**2+k**2*(m/Mpip)**2+2*q*k*(m/Mpip)*np.cos(theta))

    def f(r):
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
    r2 = np.logspace(start,stop,num=3000,base=np.exp(1))
    E = -2

    u = [0*r2,0*r2,E*r2/r2[-1]]
    res = solve_bvp(sys,bc,r2,u,p=[E],tol=1e-7,max_nodes=100000)

    phi = res.y.T[:np.size(r2),0]
    phi3 = Spline(r2,phi)

    def eta(S):
        return -charge2*mu/(hbarc**2*S)

    def F(S):
        func = lambda r: phi3(r)*r**3*2*2*np.exp(-np.pi*eta(S)/2)*abs(sp.special.gamma(1+1+1.j*S))/(sp.special.factorial(2+1))
        integral =  4*np.pi/3*sp.integrate.quad(func,0,rmax)[0]
        return integral

    return 2*10000*charge2/4/np.pi*mu/mp**2*q**3/k*np.sin(theta)**2*s**2*F(s)**2

def totalcross(Egamma,S,b):
    func = lambda theta: 2*np.pi*np.sin(theta)*diffcross(Egamma,S,b,theta)
    integ = sp.integrate.quad(func,0,np.pi)[0]
    return integ

plt.figure(figsize=(9,5.5));

photonenergies = np.linspace(145.4,180,50)


N = []
# M = []
# P = []
for i in tqdm(photonenergies):
    N.append(totalcross(i,86.2,3.8))
    # M.append(totalcross(i,100,2))
    # P.append(totalcross(i,40,3))

plt.plot(photonenergies,N, label=r'$S=86.2$ MeV, $b=3.8$ fm', color='r')
# plt.plot(photonenergies,M, label=r'$S=45.5$ MeV, $b=3.9$ fm', color='g')
# plt.plot(photonenergies,P, label=r'$S=35.4$ MeV, $b=4.0$ fm', color='navy')

plt.xlabel(r"$E_\gamma$ [MeV]");
plt.ylabel(r"$\sigma [\mu b]$");
plt.legend(loc='best',frameon=False)
plt.grid()

plt.show()
