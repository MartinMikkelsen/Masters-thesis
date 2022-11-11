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
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.integrate import quad
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


m = 139.57  #MeV
mn = 939.565378  #MeV
mu = m*mn/(mn+m) #Reduced mass
M = m+mn
g = (2*mu)
hbarc = 197.3 #MeV fm
charge2 = hbarc/(137)

def phifunc(S,b):
    def f(r):
        return S/b*np.exp(-r**2/b**2)

    def sys(r,u,E):
        y,v,I = u
        dy = v
        dv = g/(hbarc**2)*(-E+m)*y-4/r*v+g/(hbarc**2)*f(r)+charge2/r*y
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
    r = np.logspace(start,stop,num=70000,base=np.exp(1))
    E = -2

    u = [0*r,0*r,E*r/r[-1]]
    res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)
    #print(res.message,", E: ",res.p[0])

    phi = res.y.T[:np.size(r),0]
    phi3 = Spline(r,phi)

    phi_func = lambda r: phi3(r)**2*r**4
    int_phi = 4*np.pi*quad(phi_func,0,rmax)[0]
    print("Norm_integral =",int_phi)
    return res.x,abs(res.y.T[:,0]),abs(res.y.T[:,1]),res.y.T[:,2], res.p[0], int_phi

plt.figure(figsize=(9,5.5))

#phifunc(69.3,4.1)

sns.lineplot(x=phifunc(19.4,3.44)[0],y=phifunc(19.4,3.44)[1]*phifunc(19.4,3.44)[0],linewidth=3.5,label=r'$\Pi=$%0.1f MeV, $C(\psi_{N\pi^-})=$%0.2f' %(phifunc(19.4,3.44)[4],phifunc(19.4,3.44)[5]),color='r')
#sns.lineplot(x=phifunc(69.3,3.6)[0],y=phifunc(69.3,3.6)[1]*phifunc(69.3,3.6)[0],linewidth=3.5,label=r'$\Pi=$%0.1f MeV, $C(\psi_{N\pi^+})=$%0.2f' %(phifunc(69.3,3.6)[4],phifunc(69.3,3.6)[5]),color='r')
#sns.lineplot(x=phifunc(58.8,4.0)[0],y=phifunc(58.8,4.0)[1]*phifunc(58.8,4.0)[0],linewidth=3.5,label=r'$\Pi=$%0.1f MeV, $C(\psi_{N\pi^+})=$%0.2f' %(phifunc(58.8,4.0)[4],phifunc(58.8,4.0)[5]),color='g')
#sns.lineplot(x=phifunc(100.30,1.98)[0],y=phifunc(100.30,1.98)[1]*phifunc(100.30,1.98)[0],linewidth=3.5,label=r'$\Pi=$%0.1f MeV, $C(\psi_{N\pi^+})=$%0.2f' %(phifunc(100.30,1.98)[4],phifunc(100.30,1.98)[5]),color='navy')
#sns.lineplot(x=phifunc(S1,b1)[0],y=-phifunc(S1,b1)[1]*phifunc(S1,b1)[0],linewidth=3.5,label=r'$S=$%0.1f MeV, $b=$%0.1f fm, $E=$%0.1f MeV' %(S1,b1,phifunc(S1,b1)[4]))
# sns.lineplot(x=phifunc(S2,b2)[0],y=-phifunc(S2,b2)[1]*phifunc(S2,b2)[0],linewidth=3.5,label=r'$S=$%0.1f MeV, $b=$%0.1f fm, $E=$%0.1f MeV' %(S2,b2,phifunc(S2,b2)[4]))
# sns.lineplot(x=phifunc(S3,b3)[0],y=-phifunc(S3,b3)[1]*phifunc(S3,b3)[0],linewidth=3.5,label=r'$S=$%0.1f MeV, $b=$%0.1f fm, $E=$%0.1f MeV' %(S3,b3,phifunc(S3,b3)[4]))
# sns.lineplot(x=phifunc(S4,b4)[0],y=-phifunc(S4,b4)[1]*phifunc(S4,b4)[0],linewidth=3.5,label=r'$S=$%0.1f MeV, $b=$%0.1f fm, $E=$%0.1f MeV' %(S4,b4,phifunc(S4,b4)[4]))

plt.ylabel(r"$r\phi(r)$ [fm$^{-3/2}$]")
#plt.title("$S=%s$ MeV, $b=%s$ fm, \n E = %.3f" %(S,b,res.p[0]), x=0.5, y=0.8)
plt.legend(loc=0,frameon=False);
plt.xlabel("r [fm]")
plt.tight_layout()
save_fig("ContributionPlotPiMinus")
#plt.show()

# phi_func = lambda r: phi3(r)**2*r**4
# int_phi = 4*np.pi*quad(phi_func,0,rmax)[0]
# print("Norm_integral =",int_phi)
