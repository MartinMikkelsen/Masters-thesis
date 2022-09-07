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
from scipy import interpolate
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

m = 134.9  #MeV
mp = 938.2  #MeV
mu = m*mp/(mp+m) #Reduced mass
M = m+mp
g = 2*mu
hbarc = 197.3 #MeV fm
alpha = 1/137
charge = hbarc/(137)
Mpip = m+mp

def diffcross(Egamma,S,b):

    y_vals = []
    F_vals = []
    for i in tqdm(range(len(Egamma))):

        Eq = Egamma[i]-0.5*Egamma[i]**2/(Mpip)-m
        k = Egamma[i]/hbarc
        q = np.sqrt(2*mu*abs(Eq))/(hbarc)
        s = lambda theta: np.sqrt(q**2+k**2*(m/Mpip)**2+2*q*k*np.cos(theta)*m/Mpip)

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
        r2 = np.logspace(start,stop,num=2500,base=np.exp(1))
        E = -2

        u = [0*r2,0*r2,E*r2/r2[-1]]
        res = solve_bvp(sys,bc,r2,u,p=[E],tol=1e-7,max_nodes=100000)

        phi = res.y.T[:np.size(r2),0]
        phi3 = Spline(np.sort(r2),abs(phi))

        front_func = lambda theta: (-4*np.cos(2*theta)+np.cos(4*theta)+3)/(2*np.sqrt(2)*np.sqrt(12*theta-8*np.sin(2*theta)+np.sin(4*theta)))
        func = lambda r,theta: np.sin(theta)**3*((phi3(r))*r**3*spherical_jn(1,s(theta)*r))**2

        front_const = 4*np.pi**2*charge*Mpip/(mp**2)*q**3/k

        int_y = front_const*integrate.dblquad(func, 0, np.pi, lambda theta: 0, lambda theta: np.pi)[0]*10e4

        #int_F = front_const*integrate.dblquad(func, 0, np.pi, lambda r: 0, lambda r: rmax)[0]**2

        y_vals.append(int_y)
        #F_vals.append(int_F)

    return y_vals

plt.figure(figsize=(9,5.5))

diffcrossAngleBecks = [11.356073211314477, 19.217970049916808, 27.371048252911816, 34.65058236272879, 42.803660565723796, 50.374376039933445, 58.23627287853578, 66.0981697171381, 73.96006655574044, 82.11314475873544, 89.6838602329451, 97.54575707154743, 105.40765391014976, 113.26955074875208, 121.13144758735442, 128.99334442595674, 137.14642262895177, 144.7171381031614, 152.57903494176372, 160.44093178036607, 168.59400998336108]
diffcrossBecks = [0.04642857142857143, 0.07976190476190477, 0.07678571428571429, 0.1005952380952381, 0.09464285714285715, 0.14642857142857144, 0.10833333333333334, 0.14047619047619048, 0.1267857142857143, 0.12202380952380953, 0.14583333333333334, 0.13273809523809524, 0.14821428571428572, 0.13630952380952382, 0.12916666666666668, 0.09166666666666667, 0.18214285714285716, 0.10476190476190478, 0.23750000000000002, 0.15833333333333335, 0.07440476190476192]
diffcrossErrorminBecks = [0.028571428571428574, 0.06071428571428572, 0.061309523809523814, 0.08452380952380953, 0.08154761904761905, 0.13095238095238096, 0.09642857142857143, 0.1285714285714286, 0.1142857142857143, 0.11011904761904763, 0.13392857142857145, 0.11964285714285715, 0.13333333333333336, 0.12142857142857144, 0.11250000000000002, 0.07678571428571429, 0.15714285714285717, 0.08154761904761905, 0.19464285714285717, 0.11190476190476191, 0.028571428571428574]
diffcrossErrormaxBecks = diffcrossErrorminBecks
errorBecks = [diffcrossErrorminBecks, diffcrossErrormaxBecks]

plt.plot()
