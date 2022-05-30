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

def energyfunction(S,b):
    m = 139.570   #MeV
    mn = 938.2  #MeV
    mu = m*mn/(mn+m) #Reduced mass
    g = (mu)

    def f(r): #form factor
        return S*np.exp(-r**2/b**2)

    def sys(r,u,E):
        y,v,I = u
        dy = v
        dv = g*(-E+m)*y-2/r*v+g*f(r)-1
        dI = f(r)*r**4*y
        return dy,dv,dI

    def bc(ua, ub,E):
        ya,va,Ia = ua
        yb,vb,Ib = ub
        return va, vb+(g*(m+abs(E)))**0.5*yb, Ia, Ib-E

    r = np.logspace(-5,0,100)*5
    E = -0.08412275189109243

    u = [0*r,0*r,E*r/r[-1]]
    res1 = solve_bvp(sys,bc,r,u,p=[E],tol=1e-6)
    #print(res1.message,", E: ",res1.p[0])

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
    res2 = solve_bvp(sys,bc,r,u,p=[E],tol=1e-6)
    #print(res2.message,", E: ",res2.p[0])

    def plot():
        plt.figure(figsize=(9,5.5))
        sns.lineplot(x=res2.x,y=res2.y.T[:,2],linewidth=3.5) #Non rel
        sns.lineplot(x=res1.x,y=res1.y.T[:,2],linewidth=3.5) #Rel
        plt.title(r"$S=10$ MeV, $b=1$ fm", x=0.5, y=0.9)
        plt.legend(r"$E_{nonrel}$ $E_{rel}$ $\phi$".split(),loc=0);
        plt.xlabel("r [fm]")
        rs = np.linspace(0,5,np.size(res2.x))
        plt.tight_layout()

    DeltaE = res1.p[0]/res2.p[0]
    print("The energy ratio is = ", DeltaE)

    return res1.x, res2.x, res1.y.T[:,2], res2.y.T[:,2]

[a1,a2,a3,a4] = energyfunction(10,1)
[b1,b2,b3,b4] = energyfunction(0.2,1)
[c1,c2,c3,c4] = energyfunction(15,1)
[d1,d2,d3,d4] = energyfunction(10,3)

fig, axs = plt.subplots(2, 2,figsize=(15,12))
axs[0, 0].plot(a1, a3,linewidth=5,linestyle='dashed')
axs[0, 0].plot(a2, a4,linewidth=3.5)
axs[0, 0].set_title(r"$S=10$ MeV, $b=1$ fm, $E_R=0.995$", x=0.5, y=0.9)
axs[0, 0].legend(r"$E_{nonrel}$ $E_{rel}$ $\phi$".split(),loc=3);
axs[0, 0].set_xlabel("r [fm]")
axs[0, 1].plot(b1, b3,linewidth=5,linestyle='dashed')
axs[0, 1].plot(b2, b4,linewidth=3.5)
axs[0, 1].set_title(r"$S=0.2$ MeV, $b=1$ fm, $E_R=0.767$", x=0.5, y=0.9)
axs[0, 1].set_xlabel("r [fm]")
axs[1, 0].plot(c1,c3,linewidth=5,linestyle='dashed')
axs[1, 0].plot(c2,c4,linewidth=3.5)
axs[1, 0].set_title(r"$S=15$ MeV, $b=1$ fm, $E_R=0.996$", x=0.5, y=0.9)
axs[1, 0].set_xlabel("r [fm]")
axs[1, 1].plot(d1, d3,linewidth=5,linestyle='dashed')
axs[1, 1].plot(d2, d4,linewidth=3.5)
axs[1, 1].set_title(r"$S=10$ MeV, $b=3$ fm, $E_R=0.997$", x=0.5, y=0.9)
axs[1, 1].set_xlabel("r [fm]");
fig.tight_layout()
save_fig("RelativisticExpansion");
