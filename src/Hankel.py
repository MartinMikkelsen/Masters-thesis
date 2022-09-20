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

    def trapzsum(s):
        r3 = np.linspace(0,rmax,2500)
        func = phi*r2**3*spherical_jn(1,s*r2)
        int = 4*np.pi/s*integrate.simpson(func,x=r2,dx=0.01)
        return int

    return 10000*charge2/4/np.pi*mu/mp**2*q**3/k*np.sin(theta)**2*s**2*F(s)**2

#plt.figure(figsize=(9,5.5));

#print("dsigma=",diffcross(155,41.5,3.9,np.pi/2))
#angles = np.linspace(0,np.pi,50)

#M = []
#for i in angles:
#    M.append(diffcross(155,41.5,3.9,i))
#plt.plot(angles,M)

def totalcross(Egamma,S,b):
    func = lambda theta: 2*np.pi*np.sin(theta)*diffcross(Egamma,S,b,theta)
    integ = quad(func,0,np.pi)[0]
    return integ

#plt.title(r"$∫ d^3 r \, |\psi_{\bar{N}\pi}|^2=%0.2f$, $E= %0.2f$"%(totalcross(totalcross,41.5,3.9)[1],totalcross(photonenergies,41.5,3.9)[2]),x=0.3, y=0.8)
plt.figure(figsize=(9,5.5));
gammaSchmidt = np.array([144.0358208955224, 145.07462686567163, 146.22089552238805, 147.40298507462686, 148.5134328358209, 149.69552238805971, 150.84179104477613, 151.95223880597015, 153.09850746268657, 154.2089552238806, 155.31940298507462, 156.53731343283582, 157.61194029850748, 158.79402985074626, 159.9044776119403, 161.01492537313433, 162.19701492537314, 163.30746268656716, 164.4179104477612, 165.6358208955224, 166.71044776119402, 167.82089552238807])
sigmaSchmidt = np.array([0.0398406374501992, 0.049800796812749, 0.11952191235059761, 0.2290836653386454, 0.3286852589641434, 0.448207171314741, 0.5677290836653386, 0.7171314741035857, 0.9760956175298805, 1.155378486055777, 1.3545816733067728, 1.593625498007968, 1.7729083665338645, 2.1115537848605577, 2.290836653386454, 2.6095617529880477, 2.958167330677291, 3.197211155378486, 3.585657370517928, 3.9840637450199203, 4.282868525896414, 4.711155378486056])
sigmaSchmidtPoint = np.array([0, 0, 0.059602649006622516, 0.1490066225165563, 0.24834437086092717, 0.3675496688741722, 0.4768211920529801, 0.6258278145695364, 0.8841059602649007, 1.0529801324503312, 1.2417218543046358, 1.4701986754966887, 1.6291390728476822, 1.947019867549669, 2.1357615894039736, 2.433774834437086, 2.76158940397351, 2.980132450331126, 3.3675496688741724, 3.7450331125827816, 4.052980132450331, 4.420529801324504])
errorSchmidtmin = np.subtract(sigmaSchmidt,sigmaSchmidtPoint)
errorSchmidtmax = errorSchmidtmin
sigmaErrorSchmidt = [errorSchmidtmin, errorSchmidtmax]
plt.errorbar(gammaSchmidt,sigmaSchmidt,yerr=sigmaErrorSchmidt,fmt="o",color='k');

photonenergies = np.linspace(gammaSchmidt[0],170,25)
N = []
M = []
P = []
for i in photonenergies:
    N.append(totalcross(i,79.7,3.8))
    M.append(totalcross(i,41.5,3.9))
    P.append(totalcross(i,29.4,4.0))

plt.plot(photonenergies,N, label=r'$S=79.7$ MeV, $b=3.8$ fm', color='r')
plt.plot(photonenergies,M, label=r'$S=41.5$ MeV, $b=3.9$ fm', color='g')
plt.plot(photonenergies,P, label=r'$S=29.4$ MeV, $b=4.0$ fm', color='b')

plt.xlabel(r"$E_\gamma$ [MeV]");
plt.ylabel(r"$\sigma [\mu b]$");
plt.legend(loc='best')

save_fig('crossfit')
plt.grid()

plt.show()


#print("sigma150=",totalcross(150,41.5,3.9))
#print("sigma150=",totalcross(150,79.7,3.8))
#print("sigma155=",totalcross(155,41.5,3.9))
#print("sigma155=",totalcross(155,79.7,3.8))
#print("sigma160=",totalcross(160,41.5,3.9))
#print("sigma160=",totalcross(160,79.7,3.8))

# plt.plot(Photonenergy,diffcross(Photonenergy,S,b)[4])
# plt.title(r"$S=%0.2f$ MeV, $b=%0.2f$, $\theta_q = \pi/2$, trapz" %(S,b), x=0.5, y=0.9);
# plt.xlabel(r"$E_\gamma$ [MeV]");
# plt.ylabel(r"$F(s)$");
# plt.grid()
# #save_fig('F(s)trapz')
#
# plt.figure(figsize=(9,5.5));
# plt.plot(Photonenergy,diffcross(Photonenergy,S,b)[5])
# plt.title(r"$4\pi/3∫ dr \, \phi(r)r^4$", x=0.5, y=0.9);
# #save_fig('phir^4int')
#
# plt.figure(figsize=(9,5.5));
# plt.plot(Photonenergy,diffcross(Photonenergy,S,b)[0])
# plt.xlabel(r"$E_\gamma$ [MeV]");
# plt.ylabel(r"$s$ [1/fm]");
# plt.title(r"$\theta_q=\pi/2$ ", x=0.3, y=0.8);
# plt.grid()
#
# #save_fig('svsEgamma')
# plt.figure(figsize=(9,5.5));
# plt.plot(diffcross(Photonenergy,S,b)[1],diffcross(Photonenergy,S,b)[3](diffcross(Photonenergy,S,b)[1]))
# plt.title("$S=%s$ MeV, $b=%s$ fm" %(S,b), x=0.5, y=0.2)
#
# plt.show()
