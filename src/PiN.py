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
    dv = g*(-E+m)*y-4/r*v+g*f(r)
    dI = f(r)*r**4*y
    return dy,dv,dI

def bc(ua, ub,E):
    ya,va,Ia = ua
    yb,vb,Ib = ub
    return va, vb+(g*(m+abs(E)))**0.5*yb, Ia, Ib-E

r = np.logspace(-5,0,100)*5
E = -0.08412275189109243

u = [0*r,0*r,E*r/r[-1]]
res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-6)
print(res.message,", E: ",res.p[0])


def inplot():
    axins = zoomed_inset_axes(ax, 2, loc=4, bbox_to_anchor = [375, 90])
    plt.plot(res.x[110:145],res.y.T[110:145,(0,1)],linewidth=2.5)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=2, loc2=1, fc="none",ec="0.5")
    plt.draw()

plt.figure(figsize=(9,5.5))
sns.lineplot(x=res.x,y=res.y.T[:,2],linewidth=3.5)
sns.lineplot(x=res.x,y=res.y.T[:,1],linewidth=3.5)
sns.lineplot(x=res.x,y=res.y.T[:,0],linewidth=3.5)
plt.title(r"$S=10$ MeV, $b=1$ fm", x=0.5, y=0.9)
plt.legend(r"$E$ $\phi'$ $\phi$".split(),loc=0);
plt.xlabel("r [fm]")
rs = np.linspace(0,5,np.size(res.x))
plt.tight_layout()
save_fig("Integralplot")

def rms_residuals():
    plt.figure()
    plt.plot(res.x[0:315],res.rms_residuals,linewidth=2.5)
    plt.grid(); plt.legend(r"RMS".split(),loc=0);
    save_fig("rms_residuals")
