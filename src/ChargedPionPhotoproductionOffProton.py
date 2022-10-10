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


import cProfile
import re
cProfile.run('re.compile("foo|bar")', 'restats')

m = 139.57039  #MeV
mn = 938.272088  #MeV
mu = m*mn/(mn+m) #Reduced mass
g = 2*mu
hbarc = 197.327 #MeV fm
alpha = 1/137
charge2 = hbarc/(137)
Mpip = m+mn

def diffcross(Egamma,S,b,theta):

    Eq = Egamma-m-0.5*Egamma**2/(Mpip)
    if Eq<0 : return 0
    k = Egamma/hbarc
    q = np.sqrt(2*mu*Eq)/(hbarc)
    s = np.sqrt(q**2+k**2*(m/Mpip)**2+2*q*k*(m/Mpip)*np.cos(theta))

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
    r2 = np.logspace(start,stop,num=3000,base=np.exp(1))
    E = -2

    u = [0*r2,0*r2,E*r2/r2[-1]]
    res = solve_bvp(sys,bc,r2,u,p=[E],tol=1e-7,max_nodes=100000)

    phi = res.y.T[:np.size(r2),0]
    phi3 = Spline(r2,phi)

    def F(S):
        func = lambda r: phi3(r)*r**3*spherical_jn(1,S*r)
        integral =  4*np.pi/s*quad(func,0,rmax)[0]
        return integral

    return 10000*charge2/2/np.pi*mu/mn**2*q**3/k*np.sin(theta)**2*s**2*F(s)**2

def totalcross(Egamma,S,b):
    tot = []
    for i in tqdm(Egamma):
        func = lambda theta: 2*np.pi*np.sin(theta)*diffcross(i,S,b,theta)
        integ = quad(func,0,np.pi)[0]
        tot.append(integ)
    return tot

plt.figure(figsize=(9,5.5));

x = np.array([154.03437815975732, 156.01617795753288, 160.02022244691608,164.994944388271, 170.0505561172902, 175.02527805864509, 179.95955510616784])
y = np.array([36.41025641025641, 43.93162393162393, 55.72649572649573,74.52991452991454, 89.05982905982906, 98.97435897435898, 84.44444444444444])
yprime = np.array([25.470085470085472, 40.85470085470086, 52.991452991452995,70.5982905982906, 83.58974358974359, 91.7948717948718, 75.8974358974359])
errorSchmidtmin = np.subtract(y,yprime)
errorSchmidtmax = errorSchmidtmin
sigmaErrorSchmidt = [errorSchmidtmin, errorSchmidtmax]
plt.errorbar(x,y,yerr=sigmaErrorSchmidt,fmt="o",label='Included');
plt.xlabel(r"$E_\gamma$ [MeV]");
plt.ylabel(r"$\sigma [\mu b]$");
plt.grid()

popt, pcov = curve_fit(totalcross,x,y, sigma=errorSchmidtmin,p0=[100,3],ftol=0.05, xtol=0.05)
print("popt=",popt)
print("Error=",np.sqrt(np.diag(pcov)))

photonenergies = np.linspace(151.4,180,25)
plt.plot(photonenergies,totalcross(photonenergies,popt[0],popt[1]), label=r'$S=%0.2f$ MeV, $b=%0.2f$ fm' %(popt[0],popt[1]), color='r')
plt.show()
