import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import root
from scipy.special import spherical_jn
from scipy.special import spherical_jn
from scipy.integrate import solve_bvp
from scipy import fft
from sympy import hankel_transform, inverse_hankel_transform
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
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
S = 10    #MeV
m0 = 135.57  #MeV
mplus = 139.97
mp = 938.272  #MeV
mn = 939.565
mu0 = m0*mp/(mp+m0) #Reduced mass
muplus = mplus*mn/(mn+mplus) #Reduced mass
g0 = (2*mu0)
gplus = (2*muplus)
hbarc = 197.3 #MeV fm

def f(r): #form factor
    return S/b*np.exp(-r**2/b**2)

def sys(r,u,E):
    y,v,z,q,I = u
    dy = v
    dz = q
    dv = g0/(hbarc**2)*(-E+m0)*y-4/r*v+g0/(hbarc**2)*f(r)
    dq = gplus/(hbarc**2)*(-E+mplus)*z-4/r*q+gplus/(hbarc**2)*f(r)
    dI = 12*np.pi*f(r)*r**4*(y+z)
    return dy,dv,dz,dq,dI

def bc(ua, ub,E):
    ya,va,za,qa,Ia = ua
    yb,vb,zb,qb,Ib = ub
    return va, vb+(g0*(m0+abs(E)))**0.5*yb, qa,qb+(gplus*(mplus+abs(E)))**0.5*zb, Ia, Ib-E

rmax = 5*b
rmin = 0.01*b
base1 = np.exp(1)
start = np.log(rmin)
stop = np.log(rmax)
r = np.logspace(start,stop,num=2*rmax,base=np.exp(1))
E = -2

u = [0*r,0*r,0*r,0*r,E*r/r[-1]]
res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)
#print(res.message,", E: ",res.p[0])

def plot():
    plt.figure(figsize=(9,5.5));
    #sns.lineplot(x=res.x,y=res.y.T[:,4]/(24*np.pi),linewidth=3.5)
    sns.lineplot(x=res.x,y=res.y.T[:,3],linewidth=3.5,linestyle='--');
    sns.lineplot(x=res.x,y=res.y.T[:,2],linewidth=3.5,linestyle='--');
    sns.lineplot(x=res.x,y=res.y.T[:,1],linewidth=3.5);
    sns.lineplot(x=res.x,y=res.y.T[:,0],linewidth=3.5);
    plt.title("$S=%s$ MeV, $b=%s$ fm, \n E = %.3f" %(S,b,res.p[0]), x=0.5, y=0.8);
    plt.legend(r"$\phi_0'$ $\phi_0$ $\phi_+'$ $\phi_+$" .split(),loc=0,frameon=False);
    plt.xlabel("r [fm]");
    plt.tight_layout();
    #save_fig("Integralplot_CoupledSystem");
    plt.show()

plot()

def wavefunction(m,mn):
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
    r = np.logspace(start,stop,num=2*rmax,base=np.exp(1))
    E = -2

    u = [0*r,0*r,E*r/r[-1]]
    res2 = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)
    #print(res.message,", E: ",res.p[0])

    sns.lineplot(x=res2.x,y=res2.y.T[:,2]/(12*np.pi),linewidth=3.5)
    sns.lineplot(x=res2.x,y=res2.y.T[:,1],linewidth=3.5)
    sns.lineplot(x=res2.x,y=res2.y.T[:,0],linewidth=3.5)
    plt.title("$S=%s$ MeV, $b=%s$ fm, \n E = %.3f" %(S,b,res1.p[0]), x=0.5, y=0.8)
    plt.legend(r"$\frac{E}{12\pi}$ $\phi'$ $\phi$".split(),loc=0,frameon=False);
    plt.xlabel("r [fm]")
    plt.tight_layout()
