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

m = 135  #MeV
mn = 939  #MeV
mu = m*mn/(mn+m) #Reduced mass
M = m+mn
g = 2*mu
hbarc = 197.3 #MeV fm
alpha = 1/137

def sigma(Egamma,S,b):

    Eq = Egamma-m
    k = Egamma/hbarc
    q = np.sqrt(2*mu*Eq)/hbarc
    s = q+mn/M*k

    N = []
    for i in s:
        N.append(F(i,S,b))
    U = sum(np.array(N))

    frontfactors = alpha*np.sqrt(2)/(2*np.pi)*np.sqrt(Eq/mn)*(mu/mn)**(3/2)

    dsigmadOmega = frontfactors*(4*np.pi)**2*1/k*(q**2-(k*q*np.cos(np.pi/2))**2/k**2)*U**2

    return 4*np.pi*dsigmadOmega

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
    r = np.logspace(start,stop,num=1000,base=np.exp(1))
    E = -2

    u = [0*r,0*r,E*r/r[-1]]
    res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)

    phi = np.array(res.y.T[0:1000,0])
    Integral = simpson(spherical_jn(1,s*r)*phi*r**3, r, dx=0.001)

    return Integral

plt.figure(figsize=(9,5.5))
gammaBeck = [144.63155397390273, 145.18244365361804, 145.73333333333332, 146.24341637010676, 146.77390272835112, 147.34519572953735, 147.87568208778174, 148.42657176749702, 148.95705812574138, 149.5079478054567, 150.058837485172, 150.6097271648873, 151.1606168446026, 151.73190984578883, 152.28279952550415, 152.85409252669038, 153.4253855278766, 153.99667852906285, 154.5679715302491, 155.15966785290627, 155.79217081850533]
sigmaBeck = [0.06203007518796992, 0.10432330827067668, 0.14661654135338345, 0.1607142857142857, 0.23966165413533833, 0.2706766917293233, 0.30451127819548873, 0.35526315789473684, 0.40037593984962405, 0.5103383458646616, 0.5328947368421052, 0.6174812030075187, 0.6513157894736842, 0.7133458646616541, 0.7894736842105263, 0.8853383458646616, 1.00093984962406, 1.1165413533834585, 1.206766917293233, 1.401315789473684, 1.5310150375939848]
errorminPointBecks = [0.06203007518796992, 0.10432330827067668, 0.14661654135338345, 0.1607142857142857, 0.21146616541353383, 0.23684210526315788, 0.2706766917293233, 0.3157894736842105, 0.3580827067669173, 0.4595864661654135, 0.47932330827067665, 0.5582706766917293, 0.5921052631578947, 0.6484962406015037, 0.724624060150376, 0.8120300751879699, 0.9248120300751879, 0.9924812030075187, 1.1137218045112782, 1.2998120300751879, 1.4210526315789473]
crossErrormin = np.subtract(sigmaBeck,errorminPointBecks)
crossErrormaxBecks = crossErrormin
sigmaerrorBecks = [crossErrormin, crossErrormaxBecks]
plt.scatter(gammaBeck,sigmaBeck);
plt.errorbar(gammaBeck,sigmaBeck,yerr=sigmaerrorBecks,fmt="o");
plt.xlabel(r"$E_\gamma$ [MeV]");
plt.ylabel(r"$\sigma$ [$\mu$ b]");

initial = [12,2]

popt, cov = curve_fit(sigma, gammaBeck, sigmaBeck, initial, crossErrormin)
print(popt)
plt.title("$S=%0.2f$ MeV, $b=%0.2f$ fm" %(popt[0],popt[1]), x=0.5, y=0.8)
plt.xlabel(r"$E_\gamma$ [MeV]")
plt.ylabel(r"$\sigma$")
plt.tight_layout()
Photonenergy = np.linspace(gammaBeck[0],155.79217081850533,1000)
plt.plot(Photonenergy,sigma(Photonenergy,popt[0],popt[1]))
plt.show()
