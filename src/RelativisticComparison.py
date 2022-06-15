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

m = 139.570  #MeV
mn = 938.2  #MeV
mu = m*mn/(mn+m) #Reduced mass
hbarc = 197.3 #MeV fm
g = (2*mu)

def relativistic(S,b):
    def f(r): #form factor
        return S/b*np.exp(-r**2/b**2)

    def sys(r,u,E):
        y,v,z,l,I = u
        dy = v
        dv = z
        dz = l
        dl = 8*mu**3*(E-m)*y/(hbarc**4)-f(r)*8*mu**3/(hbarc**4)+4*mu**2*z/(hbarc**2)+(16*mu**2)*v/(r*(hbarc**2))-(6*l)/(r)
        dI = 12*np.pi*f(r)*r**4*y
        return dy,dv,dz,dl,dI

    def bc(ua, ub,E):
        ya,va,za,la,Ia = ua
        yb,vb,zb,lb,Ib = ub
        return va, vb,la,lb-8*mu**3*(E-m)*yb+4*mu**2*zb, Ia, Ib-E,

    rmax = 5*b
    rmin = 0.01*b
    base1 = np.exp(1)
    start = np.log(rmin)
    stop = np.log(rmax)
    r = np.logspace(start,stop,num=20*rmax,base=np.exp(1))
    E = -2

    u = [0*r,0*r,0*r,0*r,E*r/r[-1]]

    res2 = solve_bvp(sys,bc,r,u,p=[E],tol=1e-3,max_nodes=100000)
    #print(res2.message,"The relativistic energy is: ",res2.p[0])
    return res2.x, res2.y.T[:,0], res2.y.T[:,1],res2.y.T[:,2],res2.p[0]

def nonrelativistic(S,b):
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
    r = np.logspace(start,stop,num=20*rmax,base=np.exp(1))
    E = -2

    u = [0*r,0*r,E*r/r[-1]]
    res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-6)
    #print(res.message,"The nonrelativistic energy is: ",res.p[0])
    return res.x, res.y.T[:,0], res.y.T[:,1],res.y.T[:,2],res.p[0]

[a1,a2,a3,a4,a5] = relativistic(10,1)
[b1,b2,b3,b4,b5] = nonrelativistic(10,1)
print('The energy ratio is:', a5/b5)
[c1,c2,c3,c4,c5] = relativistic(10,2)
[d1,d2,d3,d4,d5] = nonrelativistic(10,2)
print('The energy ratio is:', c5/d5)
[e1,e2,e3,e4,e5] = relativistic(15,1)
[f1,f2,f3,f4,f5] = nonrelativistic(15,1)
print('The energy ratio is:', e5/f5)
[g1,g2,g3,g4,g5] = relativistic(15,2)
[h1,h2,h3,h4,h5] = nonrelativistic(15,2)
print('The energy ratio is:', g5/h5)

fig, axs = plt.subplots(2, 2,figsize=(15,12))
axs[0, 0].plot(a1, a2,linewidth=3.5,linestyle='dashed', color='g')
axs[0, 0].plot(b1, b2,linewidth=3.5, color='g')
axs[0, 0].plot(a1, a3,linewidth=3.5,linestyle='dashed', color='b')
axs[0, 0].plot(b1, b3,linewidth=3.5, color='b')
axs[0, 0].set_title(r"$S=10$ MeV, $b=1$ fm, $E_R=%.3f$" %(a5/b5), x=0.5, y=0.9)
axs[0, 0].legend(r"$\phi_{rel}$ $\phi_{nonrel}$ $\phi'_{rel}$ $\phi'_{nonrel}$".split(),loc=4, frameon=False);
axs[0, 0].set_xlabel("r [fm]")
axs[0, 0].set_ylim([-0.012,0.0075])
axs[0, 1].plot(c1, c2,linewidth=3.5,linestyle='dashed',color='g')
axs[0, 1].plot(d1, d2,linewidth=3.5,color='g')
axs[0, 1].plot(c1, c3,linewidth=3.5,linestyle='dashed', color='b')
axs[0, 1].plot(d1, d3,linewidth=3.5, color='b')
axs[0, 1].set_title(r"$S=10$ MeV, $b=2$ fm, $E_R=%.3f$" %(c5/d5), x=0.5, y=0.9)
axs[0, 1].set_xlabel("r [fm]")
axs[0, 1].set_ylim([-0.012,0.0075])
axs[0, 1].legend(r"$\phi_{rel}$ $\phi_{nonrel}$ $\phi'_{rel}$ $\phi'_{nonrel}$".split(),loc=4, frameon=False);
axs[1, 0].plot(e1, e2,linewidth=3.5,linestyle='dashed',color='g')
axs[1, 0].plot(f1, f2,linewidth=3.5, color='g')
axs[1, 0].plot(e1, e3,linewidth=3.5,linestyle='dashed', color='b')
axs[1, 0].plot(f1, f3,linewidth=3.5, color='b')
axs[1, 0].set_title(r"$S=15$ MeV, $b=1$ fm, $E_R=%.3f$" %(e5/f5), x=0.5, y=0.9)
axs[1, 0].set_xlabel("r [fm]")
axs[1, 0].set_ylim([-0.02,0.012])
axs[1, 0].legend(r"$\phi_{rel}$ $\phi_{nonrel}$ $\phi'_{rel}$ $\phi'_{nonrel}$".split(),loc=4, frameon=False);
axs[1, 1].plot(g1, g2,linewidth=3.5,linestyle='dashed',color='g')
axs[1, 1].plot(h1, h2,linewidth=3.5,color='g')
axs[1, 1].plot(g1, g3,linewidth=3.5,linestyle='dashed', color='b')
axs[1, 1].plot(h1, h3,linewidth=3.5, color='b')
axs[1, 1].set_title(r"$S=15$ MeV, $b=2$ fm, $E_R=%.3f$" %(g5/h5), x=0.5, y=0.9)
axs[1, 1].set_xlabel("r [fm]");
axs[1, 1].set_ylim([-0.02,0.012])
axs[1, 1].legend(r"$\phi_{rel}$ $\phi_{nonrel}$ $\phi'_{rel}$ $\phi'_{nonrel}$".split(),loc=4, frameon=False);
fig.tight_layout()
save_fig("RelativisticExpansion");

# plt.figure(figsize=(9,5.5))
# plt.plot(a1, a2,linewidth=3.5,linestyle='dashed', color='g')
# plt.plot(b1, b2,linewidth=3.5, color='g')
# plt.plot(a1, a3,linewidth=3.5,linestyle='dashed', color='b')
# plt.plot(b1, b3,linewidth=3.5, color='b')
# plt.title("$S=%s$ MeV, $b=%s$ fm, \n E = %.3f" %(10,1,a5/b5), x=0.5, y=0.8)
# plt.legend(r"$\phi_{rel}$ $\phi_{nonrel}$ $\phi'_{rel}$ $\phi'_{nonrel}$".split(),loc=4, frameon=False);
# plt.xlabel("r [fm]")
# plt.tight_layout()
# save_fig("Single_comparision");
