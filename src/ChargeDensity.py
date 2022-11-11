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
import scipy as sp
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

m = 139.57039  #MeV
mn = 939.565378  #MeV
mu = m*mn/(mn+m) #Reduced mass
M = m+mn
g = (2*mu)
hbarc = 197.3 #MeV fm
charge2 = hbarc/(137)

def chargedensity(S,b):
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

    rmax = 10*b
    rmin = 0.01*b
    base1 = np.exp(1)
    start = np.log(rmin)
    stop = np.log(rmax)
    r = np.logspace(start,stop,num=75000,base=np.exp(1))
    E = -2

    u = [0*r,0*r,E*r/r[-1]]
    res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)

    R = 0.0
    phi = res.y.T[:np.size(r),0]

    r_pi = R+mn/M*r
    r_N = R-m/M*r
    r_cm = (m*r_pi-mn*r_N)/(M)

    phi3 = Spline(r,phi)

    integraltest1 = quad(lambda r_cm: abs(M / mn* r_cm* phi3(M / mn * r_cm))**2,0,rmax)[0]
    print("Normalized integral is=",quad(lambda r_cm: 0.5/integraltest1*abs(M / mn* r_cm* phi3(M / mn * r_cm))**2,0,rmax)[0])
    integraltest2 = quad(lambda r_cm: abs(M /m * r_cm* phi3(M / m * r_cm))**2,0,rmax)[0]
    print("Normalized integral2 is=",quad(lambda r_cm: 0.5/integraltest2*abs(M /m * r_cm* phi3(M / m * r_cm))**2,0,rmax)[0])

    plt.figure(figsize=(9,5.5));
    plt.plot(r_cm,0.5/integraltest1*abs(M/mn*r_cm*phi3(M/mn*r_cm))**2,label=r'$\rho_{\pi}$',linewidth=2.5,color='darkgreen')
    plt.plot(r_cm,0.5/integraltest2*abs(M/m*r_cm*phi3(M/m*r_cm))**2,label=r'$\rho_p$',linewidth=2.5,color='r')
    plt.fill_between(r_cm,0.5/integraltest1*abs(M/mn*r_cm*phi3(M/mn*r_cm))**2+0.5/integraltest2*abs(M/m*r_cm*phi3(M/m*r_cm))**2,y2=0,label=r'$Q$',alpha=0.75,color='b')
    plt.title("$S=%s$ MeV, $b=%s$ fm" %(S,b), x=0.5, y=0.8)
    plt.legend(frameon=False);
    plt.xlabel(r"$|r_{cm}|$ [fm]");
    plt.ylabel(r"$\rho(r_{cm})$");
    #save_fig("ChargeDensityNeutronMinus")
    #plt.xlim([0,6])

chargedensity(63.3,3.6)
#save_fig("ChargeDensityNeutronPlus1")
chargedensity(100.30,1.98)
#save_fig("ChargeDensityNeutronPlus2")
