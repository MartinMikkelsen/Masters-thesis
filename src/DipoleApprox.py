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

m = 134.976  #MeV
mp = 938.272  #MeV
mu = m*mp/(mp+m) #Reduced mass
g = 2*mu
hbarc = 197.327 #MeV fm
alpha = 1/137
charge2 = hbarc/(137)
Mpip = m+mp

def diffcross(Egamma,S,b):

    Eq = Egamma-m-0.5*Egamma**2/(Mpip)
    if Eq<0 : return 0
    k = Egamma/hbarc
    q = np.sqrt(2*mu*Eq)/(hbarc)

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
    r2 = np.logspace(start,stop,num=10000,base=np.exp(1))
    E = -2

    u = [0*r2,0*r2,E*r2/r2[-1]]
    res = solve_bvp(sys,bc,r2,u,p=[E],tol=1e-7,max_nodes=100000)

    phi = res.y.T[:np.size(r2),0]
    phi3 = Spline(r2,phi)

    def Q(S):
        func = lambda r: phi3(r)*r**4*spherical_jn(0,S*r)
        integral =  quad(func,0,rmax)[0]
        return integral

    norm_func = lambda r: r**4*phi3(r)**2
    norm_integral = 4*np.pi*quad(norm_func,0,rmax)[0]
    N = 1/np.sqrt(1+norm_integral)
    return 8*np.pi*N**2*alpha/9*mu**3*k*q/(m**2*hbarc)*Q(q)**2

def totalcross(Egamma,S,b):
    totalX = 4*np.pi*diffcross(Egamma,S,b)
    return totalX

plt.figure(figsize=(9,5.5));

photonenergies = np.linspace(gammaSchmidt[0],151,50)
N = []
M = list()
P = []
for i in tqdm(photonenergies):
    N.append(totalcross(i,41.5,3.9))
    #M.append(totalcross(i,10,4))
    #P.append(totalcross(i,10,4))

plt.plot(photonenergies,N, label=r'$S=79.7$ MeV, $b=3.8$ fm', color='r')
plt.plot(photonenergies,M, label=r'$S=41.5$ MeV, $b=3.9$ fm', color='g')
plt.plot(photonenergies,P, label=r'$S=29.4$ MeV, $b=4.0$ fm', color='b')

gammaSchmidt = np.array([144.0358208955224, 145.07462686567163, 146.22089552238805, 147.40298507462686, 148.5134328358209, 149.69552238805971, 150.84179104477613])
sigmaSchmidt = np.array([0.0398406374501992, 0.049800796812749, 0.11952191235059761, 0.2290836653386454, 0.3286852589641434, 0.448207171314741, 0.5677290836653386])
sigmaSchmidtPoint = np.array([0, 0, 0.059602649006622516, 0.1490066225165563, 0.24834437086092717, 0.3675496688741722, 0.4768211920529801])
errorSchmidtmin = np.subtract(sigmaSchmidt,sigmaSchmidtPoint)
errorSchmidtmax = errorSchmidtmin
sigmaErrorSchmidt = [errorSchmidtmin, errorSchmidtmax]
plt.errorbar(gammaSchmidt,sigmaSchmidt,yerr=sigmaErrorSchmidt,fmt="o");

plt.xlabel(r"$E_\gamma$ [MeV]");
plt.ylabel(r"$\sigma [\mu b]$");
plt.legend(loc='best',frameon=False)
plt.grid()
plt.show()
