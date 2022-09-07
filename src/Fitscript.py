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
    phi_vals = []
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

        phi_func = lambda r: phi3(r)**2*r**2
        int_phi = 12*np.pi*quad(phi_func,0,rmax)[0]

        y_vals.append(int_y)
        phi_vals.append(int_phi)

    return y_vals, int_phi, res.p[0]


plt.figure(figsize=(9,5.5))

gammaSchmidt = np.array([144.0358208955224, 145.07462686567163, 146.22089552238805, 147.40298507462686, 148.5134328358209, 149.69552238805971, 150.84179104477613, 151.95223880597015, 153.09850746268657, 154.2089552238806, 155.31940298507462, 156.53731343283582, 157.61194029850748, 158.79402985074626, 159.9044776119403, 161.01492537313433, 162.19701492537314, 163.30746268656716, 164.4179104477612, 165.6358208955224, 166.71044776119402, 167.82089552238807])
sigmaSchmidt = np.array([0.0398406374501992, 0.049800796812749, 0.11952191235059761, 0.2290836653386454, 0.3286852589641434, 0.448207171314741, 0.5677290836653386, 0.7171314741035857, 0.9760956175298805, 1.155378486055777, 1.3545816733067728, 1.593625498007968, 1.7729083665338645, 2.1115537848605577, 2.290836653386454, 2.6095617529880477, 2.958167330677291, 3.197211155378486, 3.585657370517928, 3.9840637450199203, 4.282868525896414, 4.711155378486056])
sigmaSchmidtPoint = np.array([0, 0, 0.059602649006622516, 0.1490066225165563, 0.24834437086092717, 0.3675496688741722, 0.4768211920529801, 0.6258278145695364, 0.8841059602649007, 1.0529801324503312, 1.2417218543046358, 1.4701986754966887, 1.6291390728476822, 1.947019867549669, 2.1357615894039736, 2.433774834437086, 2.76158940397351, 2.980132450331126, 3.3675496688741724, 3.7450331125827816, 4.052980132450331, 4.420529801324504])
errorSchmidtmin = np.subtract(sigmaSchmidt,sigmaSchmidtPoint)
errorSchmidtmax = errorSchmidtmin
sigmaErrorSchmidt = [errorSchmidtmin, errorSchmidtmax]
sigmaErrorTest = np.subtract(errorSchmidtmax,errorSchmidtmin)



def cross(Egamma,S,b):
    cross_section = np.array(diffcross(gammaSchmidt,S,3.9)[0])
    return cross_section

plt.errorbar(gammaSchmidt,sigmaSchmidt,yerr=sigmaErrorSchmidt,fmt="o");
plt.xlabel(r"$E_\gamma$ [MeV]");
plt.ylabel(r"$\sigma$ [$\mu$b]");
Photonenergy = np.linspace(gammaSchmidt[0],gammaSchmidt[21],22)
#plt.plot(Photonenergy,cross(Photonenergy,144.17,3.61),'--')


#plt.plot(Photonenergy,cross(Photonenergy,85,3.8))
#plt.plot(Photonenergy,cross(Photonenergy,45,3.9))
#plt.plot(Photonenergy,cross(Photonenergy,33.4,4))

#plt.title("$S=80$ MeV, $b=3.7$ fm")
initial = [41.5,3.8]
popt, cov = curve_fit(cross, gammaSchmidt, sigmaSchmidt, p0=initial, sigma=errorSchmidtmax)
plt.title("$S=%0.2f$ MeV, $b=%0.2f$ fm, fit" %(popt[0],popt[1]), x=0.5, y=0.8)
plt.xlabel(r"$E_\gamma$ [MeV]")
plt.ylabel(r"$\sigma$ [$\mu$b]")
plt.tight_layout()
plt.plot(Photonenergy,cross(Photonenergy,popt[0],popt[1]))
save_fig("fit")
print("The norm of the wave function is =", diffcross(Photonenergy,popt[0],popt[1])[1])
print("The energy is =", diffcross(Photonenergy,popt[0],popt[1])[2])
plt.show()
