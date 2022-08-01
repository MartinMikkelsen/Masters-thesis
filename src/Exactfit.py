import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.integrate import quad
from scipy.optimize import root
from scipy.integrate import simpson
from scipy.integrate import solve_bvp
from scipy.special import spherical_jn
from scipy.optimize import curve_fit
import seaborn as sns
import os
from scipy import integrate
from pylab import plt, mpl
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from hankel import HankelTransform     # Import the basic class

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

m = 135  #MeV
mn = 939  #MeV
mp = 938.927  #MeV
mu = m*mn/(mn+m) #Reduced mass
M = m+mn
g = 2*mu
hbarc = 197.3 #MeV fm
alpha = 1/137

def sigma(Egamma,S,b):

    Eq = Egamma-m
    k = Egamma/hbarc
    q = np.sqrt(2*mu*Eq)/hbarc
    s = q+mp/M*k
    frontfactors = np.sqrt(2)*8*np.pi**3*alpha*(mu/m)**(3/2)

    CrossSection = frontfactors*np.sqrt(Eq/m)*(q**2/k)*F(s,S,b)**2

    return CrossSection*10e6

def F(s,S,b):

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
    r = np.logspace(start,stop,num=10,base=np.exp(1))
    E = -2

    u = [0*r,0*r,E*r/r[-1]]
    res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)

    phi = res.y.T[:10,0]

    # j_l(z) = √\frac{π}{2z} J_{l+1/2}(z)

    func = Spline(r,phi*r**2*np.sqrt(np.pi/(2*s*r)))

    ht = HankelTransform(
        nu= 3/2,     # The order of the bessel function
        N = 120,     # Number of steps in the integration
        h = 0.03     # Proxy for "size" of steps in integration
    )
    Fs = ht.transform(func,s,ret_err=False) # Return the transform of f at s.

    return Fs

plt.figure(figsize=(9,5.5))

gammaFuchs = np.array([145.29, 146.11, 146.99, 147.82, 148.97, 149.83, 150.86, 151.69, 152.53, 153.37])
sigmaFuchs = np.array([0.056, 0.112, 0.158, 0.202, 0.284, 0.390, 0.462, 0.589, 0.676, 0.801])
errorFuchsmin = np.array([0.009, 0.011, 0.009, 0.014, 0.016, 0.017, 0.019, 0.026, 0.024, 0.027])
errorFuchsmax = errorFuchsmin
errorFuchs = [errorFuchsmin, errorFuchsmax]
plt.errorbar(gammaFuchs,sigmaFuchs,yerr=errorFuchs,fmt="o");
plt.xlabel(r"$E_\gamma$ [MeV]")
plt.ylabel(r"$\sigma$ [$\mu$b]")

initial = [12.25,1.456]

popt, cov = curve_fit(sigma, gammaFuchs, sigmaFuchs, initial, errorFuchsmax)
print(popt)
plt.title("$S=%0.2f$ MeV, $b=%0.2f$ fm" %(popt[0],popt[1]), x=0.5, y=0.8)
plt.xlabel(r"$E_\gamma$ [MeV]")
plt.ylabel(r"$\sigma$")
plt.tight_layout()
Photonenergy = np.linspace(gammaFuchs[0],gammaFuchs[9],10)
plt.plot(Photonenergy,sigma(Photonenergy,popt[0],popt[1]))
#save_fig("fit")
