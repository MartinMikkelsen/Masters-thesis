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
    dp2dEq = ((Eq**2+2*Eq*mp+2*mp**2+2*Eq*m+2*mp*m)*(Eq**2+2*Eq*mp+2*m**2+2*Eq*m+2*mp*m))/(2*(Eq+mp+m)**3)

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

    return 10000*charge2/8/np.pi*dp2dEq/mp**2*q**3/k*np.sin(theta)**2*s**2*F(s)**2

def PlotC(EgammaPlot,S,b):
    angles = np.linspace(0,np.pi,150)
    M = []
    for i in angles:
       M.append(diffcross(EgammaPlot,S,b,i))
    plt.figure(figsize=(9,5.5))
    plt.plot(np.rad2deg(angles),M, label=r'$E_\gamma=%0.1f$ MeV, $S=%0.1f$ MeV, $b=%0.1f$ fm' %(EgammaPlot,S,b),color='r')
    AngleBeck = np.array([12.542955326460481, 20.61855670103093, 28.694158075601376, 36.597938144329895, 44.50171821305842, 52.40549828178694, 60.48109965635739, 68.21305841924399, 76.11683848797252, 84.02061855670104, 92.09621993127148, 100.34364261168385, 108.07560137457045, 116.1512027491409, 124.22680412371135, 132.3024054982818, 148.28178694158075, 156.18556701030928, 164.26116838487974])
    DiffCrossBeck = np.array([0.009547738693467337, 0.022738693467336684, 0.033668341708542715, 0.03618090452261307, 0.019472361809045227, 0.05188442211055277, 0.04082914572864322, 0.06733668341708543, 0.05841708542713568, 0.0742462311557789, 0.0678391959798995, 0.05653266331658292, 0.07763819095477388, 0.05326633165829146, 0.05515075376884423, 0.05276381909547739, 0.028517587939698497, 0.04396984924623116, 0.03869346733668342])
    Err = np.array([0.0023869346733668344, 0.013316582914572866, 0.023994974874371862, 0.02726130653266332, 0.013442211055276383, 0.042713567839195984, 0.0335427135678392, 0.05804020100502513, 0.050125628140703524, 0.06482412060301508, 0.05879396984924624, 0.047989949748743724, 0.06733668341708543, 0.04396984924623116, 0.04459798994974875, 0.04133165829145729, 0.017713567839195983, 0.027889447236180906, 0.019723618090452262])
    Errormin = np.subtract(DiffCrossBeck,Err)
    Errormax = Errormin
    ErrorBeck = [Errormin, Errormax]
    plt.errorbar(AngleBeck,DiffCrossBeck,yerr=ErrorBeck,fmt="o",label=r'$E_\gamma=151.4$ MeV');
    plt.xlabel(r"$\theta_q$ [deg]");
    plt.ylabel(r"$d\sigma$/d$\Omega_q$ [$\mu$b/sr]");
    plt.tight_layout()
    plt.plot(np.rad2deg(angles),0.06*np.sin(angles)**2,'--', label=r'$0.06\sin(\theta)^2$')
    plt.legend(loc='upper left',frameon=False)
    #plt.grid()
    save_fig("DiffCross151_rel")


PlotC(151.4,86.2,3.8)
