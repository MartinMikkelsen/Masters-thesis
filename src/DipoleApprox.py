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

def DipolePiPlus(b,S):
    m = 139  #MeV
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
    r = np.logspace(start,stop,num=750,base=np.exp(1))
    E = -2

    u = [0*r,0*r,E*r/r[-1]]
    res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)

    V = 1
    intphi = 1#3*V*np.trapz(res.y.T[:,0]**2*r**2, r,dx=0.001)
    N = 1#1/np.sqrt(V)*1/(np.sqrt(1+intphi))
    alpha = 1/(137)

    frontfactors = 16/9*np.pi*alpha*N**2*np.sqrt(2)
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

    dsigmadomega = frontfactors*np.sqrt(Eq/m)*(Eq+m)/(hbarc)**3*mu**2*U**2
    return dsigmadomega

def DipolePi0(b,S):
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
    r = np.logspace(start,stop,num=750,base=np.exp(1))
    E = -2

    u = [0*r,0*r,E*r/r[-1]]
    res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)

    V = 1
    intphi = 0#3*V*np.trapz(res.y.T[:,0]**2*r**2, r,dx=0.001)
    N = 1#1/np.sqrt(V)*1/(np.sqrt(1+intphi))
    alpha = 1/(137)

    frontfactors = 16/(2*9)*np.pi*alpha*N**2*np.sqrt(2)
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

    dsigmadomega = frontfactors*np.sqrt(Eq/m)*(Eq+m)/(hbarc)**3*mu**2*U**2
    return dsigmadomega

Egamma1 = np.linspace(139,150,np.size(res.y.T[:,0]))
Egamma2 = np.linspace(135,150,np.size(res.y.T[:,0]))

b = 0.65
S = 12
plt.figure(figsize=(9,5.5))
plt.title("$S=%s$ MeV, $b=%s$ fm" %(S,b), x=0.5, y=0.8)
plt.xlabel(r"$E_\gamma$ [MeV]")
plt.ylabel(r"$\sigma$")
plt.tight_layout()
plt.plot(Egamma1+10,4*np.pi*DipolePiPlus(b,S),linewidth=3.5);
plt.plot(Egamma2+10,4*np.pi*DipolePi0(b,S),linewidth=3.5);

#Comparing

gammaFuchs = [145.29, 146.11, 146.99, 147.82, 148.97, 149.83, 150.86, 151.69, 152.53, 153.37]
sigmaFuchs = [0.056, 0.112, 0.158, 0.202, 0.284, 0.390, 0.462, 0.589, 0.676, 0.801]
errorFuchsmin = [0.009, 0.011, 0.009, 0.014, 0.016, 0.017, 0.019, 0.026, 0.024, 0.027]
errorFuchsmax = errorFuchsmin
errorFuchs = [errorFuchsmin, errorFuchsmax]
plt.scatter(gammaFuchs,sigmaFuchs);
plt.errorbar(gammaFuchs,sigmaFuchs,yerr=errorFuchs,fmt="o");
plt.xlabel(r"$E_\gamma$ [GeV]");
plt.ylabel(r"$\sigma$ [$\mu$b]");
plt.axvline(x=135+10, color='r', linestyle='--')
plt.axvline(x=139+10, color='r', linestyle='--')
plt.legend(r"$(\gamma,\pi^+)$ $(\gamma,\pi^0)$".split(),loc=0,frameon=False);
save_fig("theorypgamma2")
