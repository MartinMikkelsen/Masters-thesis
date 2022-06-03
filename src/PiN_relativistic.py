import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import root
from scipy.special import spherical_jn
from scipy.integrate import solve_bvp
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

b = 1
S = 10
m = 139.570   #MeV
mn = 938.2  #MeV
mu = m*mn/(mn+m) #Reduced mass

def f(r): #form factor
    return S*np.exp(-r**2/b**2)

def sys(r,u,E):
    y,v,z,l,I = u
    dy = v
    dv = z
    dz = l
    dl = 8*mu**3*(E-m)*y-f(r)*8*mu**3+4*mu**2*z-(16*mu**2)*v/(r)-(24*mu**2*l)/r
    dI = f(r)*r**4*y
    return dy,dv,dz,dl,dI

def bc(ua, ub,E):
    ya,va,za,la,Ia = ua
    yb,vb,zb,lb,Ib = ub
    return va, vb,la,lb-8*mu**3*(E-m)*yb+4*mu**2*zb, Ia, Ib-E,

r = np.logspace(-5,0,100)*5
E = -2

u = [0*r,0*r,0*r,0*r,E*r/r[-1]]

res2 = solve_bvp(sys,bc,r,u,p=[E],tol=1e-3,max_nodes=100000)
print(res.message,", E: ",res2.p[0])

plt.figure(figsize=(9,5.5))
sns.lineplot(x=res2.x,y=res2.y.T[:,4],linewidth=3.5) #Energy
sns.lineplot(x=res2.x,y=res2.y.T[:,1],linewidth=3.5) #1st dv
sns.lineplot(x=res2.x,y=res2.y.T[:,0],linewidth=3.5)
sns.lineplot(x=res2.x,y=res2.y.T[:,3],linewidth=2,linestyle='--') #3rd dv
sns.lineplot(x=res2.x,y=res2.y.T[:,2],linewidth=2,linestyle='--') #2nd dv
#plt.ylim([-0.08,0.06])
plt.title(r"$S=10$ MeV, $b=1$ fm", x=0.5, y=0.9)
plt.legend(r"$E$ $\phi'$ $\phi$ $\phi'''$ $\phi''$".split(),loc=0);
plt.xlabel("r [fm]")
rs = np.linspace(0,5,np.size(res.x))
plt.tight_layout()
save_fig("Integralplot_relativistic")
