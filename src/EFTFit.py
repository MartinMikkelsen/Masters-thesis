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
from scipy.optimize import curve_fit
from tqdm import tqdm
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

def diffcross(Egamma,S,b,theta):

    m = 134.976  #MeV
    m = 135.57  #MeV
    mp = 938.272  #MeV
    mu = m*mp/(mp+m) #Reduced mass
    g = 2*mu
    hbarc = 197.327 #MeV fm
    alpha = 1/137
    charge2 = hbarc/(137)
    Mpip = m+mp

    Eq = np.array(Egamma-m-0.5*Egamma**2/(Mpip))
    if Eq<0 : return 0
    k = np.array(Egamma/hbarc)
    q = np.array(np.sqrt(2*mu*Eq)/(hbarc))
    s = np.array(np.sqrt(q**2+k**2*(m/Mpip)**2+2*q*k*(m/Mpip)*np.cos(theta)))

    def f(r): #form factor
        return S/b*np.exp(-r**2/b**2)

    def df(r): #d/dr f(r)
        return -2*r/b**2*S/b*np.exp(-r**2/b**2)

    def ddf(r): #d^2/dr^2 f(r)
        return -2/b**4*(b**2-2*r**2)*S/b*np.exp(-r**2/b**2)

    def sys(r,u,E):
        y,v,z,I = u
        dy = v
        dv = z
        dz = mu/(2*hbarc**2)*(-E-m)*v-mu/(hbarc**2)*2*r/b**2*f(r)
        dI = 12*np.pi*(2*f(r)*y+r**2*y+2*r*f(r)*v+2*r*df(r)*y+r**2*ddf(r)*y+r**2*df(r)*v+2*r*f(r)*y+r**2*df(r)*v+r**2*f(r)*z)
        return dy,dv,dz,dI

    def bc(ua, ub, E):
        ya,va,za,Ia = ua
        yb,vb,zb,Ib = ub
        return va, vb+(g*(m+abs(E)))**0.5*yb,yb, Ia, Ib-E

    rmax = 5*b
    rmin = 0.01*b
    base1 = np.exp(1)
    start = np.log(rmin)
    stop = np.log(rmax)
    r = np.logspace(start,stop,num=50000,base=np.exp(1))
    E = -2

    u = [0*r,0*r,0*r,E*r/r[-1]]
    res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)

    phi = res.y.T[:np.size(r),0]
    phi3 = Spline(r,phi)

    def F(S):
        func = lambda r: phi3(r)*r**3*spherical_jn(1,S*r)
        integral =  4*np.pi/s*quad(func,0,rmax)[0]
        return integral

    return np.array(10000*charge2/4/np.pi*mu/mp**2*q**3/k*np.sin(theta)**2*s**2*F(s)**2)

def totalcross(Egamma,S,b):
    tot = [quad(lambda theta: 2*np.pi*np.sin(theta)*diffcross(i,S,b,theta),0,np.pi)[0] for i in tqdm(Egamma)]
    return tot

if __name__ == '__main__':
    plt.figure(figsize=(9,5.5));
    gammaSchmidt = np.array([144.0358208955224, 145.07462686567163, 146.22089552238805, 147.40298507462686, 148.5134328358209, 149.69552238805971, 150.84179104477613, 151.95223880597015, 153.09850746268657, 154.2089552238806, 155.31940298507462, 156.53731343283582, 157.61194029850748, 158.79402985074626, 159.9044776119403, 161.01492537313433, 162.19701492537314, 163.30746268656716, 164.4179104477612, 165.6358208955224, 166.71044776119402, 167.82089552238807])
    sigmaSchmidt = np.array([0.0398406374501992, 0.049800796812749, 0.11952191235059761, 0.2290836653386454, 0.3286852589641434, 0.448207171314741, 0.5677290836653386, 0.7171314741035857, 0.9760956175298805, 1.155378486055777, 1.3545816733067728, 1.593625498007968, 1.7729083665338645, 2.1115537848605577, 2.290836653386454, 2.6095617529880477, 2.958167330677291, 3.197211155378486, 3.585657370517928, 3.9840637450199203, 4.282868525896414, 4.711155378486056])
    sigmaSchmidtPoint = np.array([0, 0, 0.059602649006622516, 0.1490066225165563, 0.24834437086092717, 0.3675496688741722, 0.4768211920529801, 0.6258278145695364, 0.8841059602649007, 1.0529801324503312, 1.2417218543046358, 1.4701986754966887, 1.6291390728476822, 1.947019867549669, 2.1357615894039736, 2.433774834437086, 2.76158940397351, 2.980132450331126, 3.3675496688741724, 3.7450331125827816, 4.052980132450331, 4.420529801324504])
    errorSchmidtmin = np.subtract(sigmaSchmidt,sigmaSchmidtPoint)
    errorSchmidtmax = errorSchmidtmin
    sigmaErrorSchmidt = [errorSchmidtmin, errorSchmidtmax]
    plt.errorbar(gammaSchmidt,sigmaSchmidt,yerr=sigmaErrorSchmidt,fmt="o",color='b')
    #popt, pcov = curve_fit(totalcross,gammaSchmidt,sigmaSchmidt, sigma=errorSchmidtmin)
    #print("popt=",popt)
    #print("Error=",np.sqrt(np.diag(pcov)))
    photonenergies = np.linspace(144.7,170,25)
    plt.plot(photonenergies,totalcross(photonenergies,75,3.9))
    plt.xlabel(r"$E_\gamma$ [MeV]");
    plt.ylabel(r"$\sigma [\mu b]$");
    plt.grid()
    plt.show()
