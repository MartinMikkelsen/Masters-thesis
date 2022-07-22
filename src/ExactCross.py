import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.integrate import quad
from scipy.optimize import root
from scipy.integrate import solve_bvp
from scipy.special import spherical_jn
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

b = 4.26    #fm
S = 12   #MeV
m = 135  #MeV
mn = 939  #MeV
mu = m*mn/(mn+m) #Reduced mass
M = m+mn
g = (2*mu)
hbarc = 197.3 #MeV fm
alpha = 1/137

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
r = np.logspace(start,stop,num=5000,base=np.exp(1))
E = -2

u = [0*r,0*r,E*r/r[-1]]
res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)

def radtodeg(x):
    degree=(x*180)/np.pi
    return degree

phi = res.y.T[:,0]

def F(s):
    Integral = np.trapz(spherical_jn(1,s*r)*phi*r**3, r, dx=0.001)
    return Integral

def dsigmadOmegaAngle(Egamma,theta):
    Eq = Egamma-m
    k = Egamma/hbarc
    q = np.sqrt(2*mu*Eq)/hbarc
    s = q+mn/M*k

    frontfactors = alpha*np.sqrt(2)/(2*np.pi)*np.sqrt(Eq/mn)*(mu/mn)**(3/2)

    dsigmadOmega = frontfactors*1/k*(q**2-(k*q*np.cos(theta))**2/k**2)*F(s)**2
    return dsigmadOmega

def sigma(Egamma):
    Eq = Egamma-m
    k = Egamma/hbarc
    q = np.sqrt(2*mu*Eq)/hbarc
    s = q+mn/M*k

    frontfactors = alpha*np.sqrt(2)/(2*np.pi)*np.sqrt(Eq/mn)*(mu/mn)**(3/2)

    dsigmadOmega = frontfactors*1/k*(q**2-(k*q*np.cos(np.pi/2))**2/k**2)*F(s)**2
    return dsigmadOmega

gammaFuchs = [145.29, 146.11, 146.99, 147.82, 148.97, 149.83, 150.86, 151.69, 152.53, 153.37]
sigmaFuchs = [0.056, 0.112, 0.158, 0.202, 0.284, 0.390, 0.462, 0.589, 0.676, 0.801]
errorFuchsmin = [0.009, 0.011, 0.009, 0.014, 0.016, 0.017, 0.019, 0.026, 0.024, 0.027]
errorFuchsmax = errorFuchsmin
errorFuchs = [errorFuchsmin, errorFuchsmax]

#plt.scatter(gammaFuchs,sigmaFuchs);
#plt.errorbar(gammaFuchs,sigmaFuchs,yerr=errorFuchs,fmt="o");
plt.figure(figsize=(9,5.5))

plt.xlabel(r"$E_\gamma$ [GeV]")
plt.ylabel(r"$\sigma$ [mb]")

theta = np.linspace(0,np.pi,np.size(res.x))
Photonenergy = np.linspace(145,153,np.size(res.x))
#plt.plot(Photonenergy,sigma(Photonenergy)*4*np.pi*10e6)
#plt.plot(Photonenergy,dsigmadOmegaAngle(Photonenergy,np.pi/2)*4*np.pi*10e6);
plt.plot(radtodeg(theta),dsigmadOmegaAngle(151.4,theta)*10e7);


diffcrossAngleBecks = [12.252475247524753, 20.495049504950494, 28.514851485148515, 36.53465346534654, 44.77722772277228, 52.351485148514854, 60.5940594059406, 68.16831683168317, 76.1881188118812, 84.20792079207921, 92.45049504950495, 100.24752475247524, 108.26732673267327, 116.2871287128713, 124.3069306930693, 132.54950495049505, 148.5891089108911, 156.83168316831683, 164.85148514851485]
diffcrossBecks = [0.009536637931034483, 0.022629310344827586, 0.033459051724137934, 0.03588362068965517, 0.01939655172413793, 0.05140086206896552, 0.04057112068965517, 0.06691810344827587, 0.05818965517241379, 0.07370689655172413, 0.06756465517241379, 0.05641163793103448, 0.07742456896551723, 0.05301724137931035, 0.05495689655172414, 0.052370689655172414, 0.028448275862068967, 0.043642241379310345, 0.038308189655172416]
diffcrossErrorminPointBecks = [0.002586206896551724, 0.013415948275862068, 0.024245689655172414, 0.027316810344827587, 0.013577586206896551, 0.04186422413793103, 0.03362068965517241, 0.05786637931034483, 0.04994612068965517, 0.06481681034482759, 0.05867456896551724, 0.04768318965517241, 0.06691810344827587, 0.04412715517241379, 0.04461206896551724, 0.04154094827586207, 0.017780172413793104, 0.027801724137931035, 0.01939655172413793]
diffcrossErrormin = np.subtract(diffcrossBecks,diffcrossErrorminPointBecks)
diffcrossErrormaxBecks = diffcrossErrormin
errorBecks = [diffcrossErrormin, diffcrossErrormaxBecks]
plt.scatter(diffcrossAngleBecks,diffcrossBecks);
plt.errorbar(diffcrossAngleBecks,diffcrossBecks,yerr=errorBecks,fmt="o");
plt.xlabel(r"$\theta$ [deg]");
plt.title("$S=%s$ MeV, $b=%s$ fm, \n $E_\gamma$ = %s" %(S,b,151.4), x=0.5, y=0.8)
plt.ylabel(r"$d\sigma/d\Omega$ [$\mu$ b/sr]");
#save_fig("AngularDependency151")
plt.show()

"""
diffcrossAngleBecks = [11.356073211314477, 19.217970049916808, 27.371048252911816, 34.65058236272879, 42.803660565723796, 50.374376039933445, 58.23627287853578, 66.0981697171381, 73.96006655574044, 82.11314475873544, 89.6838602329451, 97.54575707154743, 105.40765391014976, 113.26955074875208, 121.13144758735442, 128.99334442595674, 137.14642262895177, 144.7171381031614, 152.57903494176372, 160.44093178036607, 168.59400998336108]
diffcrossBecks = [0.04642857142857143, 0.07976190476190477, 0.07678571428571429, 0.1005952380952381, 0.09464285714285715, 0.14642857142857144, 0.10833333333333334, 0.14047619047619048, 0.1267857142857143, 0.12202380952380953, 0.14583333333333334, 0.13273809523809524, 0.14821428571428572, 0.13630952380952382, 0.12916666666666668, 0.09166666666666667, 0.18214285714285716, 0.10476190476190478, 0.23750000000000002, 0.15833333333333335, 0.07440476190476192]
diffcrossErrorminPointBecks = [0.028571428571428574, 0.06071428571428572, 0.061309523809523814, 0.08452380952380953, 0.08154761904761905, 0.13095238095238096, 0.09642857142857143, 0.1285714285714286, 0.1142857142857143, 0.11011904761904763, 0.13392857142857145, 0.11964285714285715, 0.13333333333333336, 0.12142857142857144, 0.11250000000000002, 0.07678571428571429, 0.15714285714285717, 0.08154761904761905, 0.19464285714285717, 0.11190476190476191, 0.028571428571428574]
diffcrossErrormin = np.subtract(diffcrossBecks,diffcrossErrorminPointBecks)
diffcrossErrormaxBecks = diffcrossErrormin
errorBecks = [diffcrossErrormin, diffcrossErrormaxBecks]
plt.scatter(diffcrossAngleBecks,diffcrossBecks);
plt.errorbar(diffcrossAngleBecks,diffcrossBecks,yerr=errorBecks,fmt="o");
plt.xlabel(r"$\theta$ [deg]");
plt.ylabel(r"$d\sigma/d\Omega$ [$\mu$ b/sr]");
plt.show()
"""
