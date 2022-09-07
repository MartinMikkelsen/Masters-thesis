import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.integrate import quad
from scipy.optimize import root
from scipy.integrate import simpson
from scipy.integrate import solve_bvp
from scipy.special import spherical_jn
from scipy.special import jv
from scipy.optimize import curve_fit
import seaborn as sns
import os
from scipy import integrate
from pylab import plt, mpl
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from hankel import HankelTransform     # Import the basic class
from tqdm import tqdm
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

m = 134.98  #MeV
mp = 938.27  #MeV
mu = m*mp/(mp+m) #Reduced mass
g = 2*mu
hbarc = 197.327 #MeV fm
alpha = 1/137
charge = hbarc/(137)
Mpip = m+mp

def diffcross(Egamma,S,b):

    y_vals = []
    Fs = []
    s_vals = []

    Eq = Egamma-m-0.5*Egamma**2/(Mpip)
    k = Egamma/hbarc
    q = np.sqrt(2*mu*abs(Eq))/(hbarc)
    s = np.sqrt(q**2+k**2*(m/Mpip)**2)

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
    r2 = np.logspace(start,stop,num=2500,base=np.exp(1))
    E = -2

    u = [0*r2,0*r2,E*r2/r2[-1]]
    res = solve_bvp(sys,bc,r2,u,p=[E],tol=1e-7,max_nodes=100000)

    phi = res.y.T[:np.size(r2),0]
    phi3 = Spline(r2,phi)

    def F(s):
        func = lambda r: phi3(r)*r**3*spherical_jn(1,s*r)
        integral =  4*np.pi/s*quad(func,0,rmax)[0]
        return integral

    def HankelF(s):
        Hankelfunc = lambda r: abs(phi3(r))*r**3*np.sqrt(np.pi/(2*s*r))*jv(3/2,s*r)
        integral = 4*np.pi/s*quad(Hankelfunc,0,rmax)[0]
        return integral

    N = []
    for i in s:
        N.append(F(i))

    U = np.array(N)

    return s,r2, U, phi3

plt.figure(figsize=(9,5.5));
S,b = [41.5,3.9]

Photonenergy = np.linspace(144.7,160,1000)
plt.plot(Photonenergy,diffcross(Photonenergy,S,b)[2])
plt.title(r"$S=%0.2f$ MeV, $b=%0.2f$, $\theta_q = \pi/2$" %(S,b), x=0.5, y=0.8);
plt.xlabel(r"$E_\gamma$ [MeV]");
plt.ylabel(r"$F(s)$");
plt.grid()
save_fig('F(s)')

plt.figure(figsize=(9,5.5));
plt.plot(Photonenergy,diffcross(Photonenergy,S,b)[0])
plt.xlabel(r"$E_\gamma$ [MeV]");
plt.ylabel(r"$s$ [1/fm]");
plt.title(r"$\theta_q=\pi/2$ ", x=0.3, y=0.8);
plt.grid()
save_fig('svsEgamma')
plt.figure(figsize=(9,5.5));

plt.plot(diffcross(Photonenergy,S,b)[1],diffcross(Photonenergy,S,b)[3](diffcross(Photonenergy,S,b)[1]))


plt.show()
