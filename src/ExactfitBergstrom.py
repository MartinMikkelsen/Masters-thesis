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
from scipy import integrate
from pylab import plt, mpl
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from hankel import HankelTransform     # Import the basic class

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
mp = 938.927  #MeV
mu = m*mn/(mn+m) #Reduced mass
M = m+mn
g = 2*mu
hbarc = 197.3 #MeV fm
alpha = 1/137

def sigma(Egamma,S,b):

    Eq = Egamma-m
    k = Egamma/hbarc
    q = np.sqrt(2*mu*Eq)/hbarc
    s = q+mp/M*k
    frontfactors = np.sqrt(2)*8*np.pi**3*alpha*(mu/m)**(3/2)

    CrossSection = frontfactors*np.sqrt(Eq/m)*(q**2/k)*F(s,S,b)**2

    return CrossSection*10e6

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
    r = np.logspace(start,stop,num=29,base=np.exp(1))
    E = -2

    u = [0*r,0*r,E*r/r[-1]]
    res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)

    phi = res.y.T[:29,0]

    # j_l(z) = √\frac{π}{2z} J_{l+1/2}(z)

    func = Spline(r,phi*r**2*np.sqrt(np.pi/(2*s*r)))

    ht = HankelTransform(
        nu= 3/2,     # The order of the bessel function
        N = 120,     # Number of steps in the integration
        h = 0.03     # Proxy for "size" of steps in integration
    )
    Fs = ht.transform(func,s,ret_err=False) # Return the transform of f at s.

    return Fs

plt.figure(figsize=(9,5.5))

gammaBergstrom = [141.55172413793105, 142.41379310344828, 143.27586206896552, 144.13793103448276, 145, 145.86206896551724, 146.72413793103448, 147.5287356321839, 148.39080459770116, 149.2528735632184, 150.05747126436782, 150.91954022988506, 151.66666666666666, 152.4712643678161, 153.27586206896552, 154.08045977011494, 154.88505747126436, 155.632183908046, 156.4367816091954, 157.29885057471265, 158.04597701149424, 158.85057471264366, 159.5977011494253, 160.3448275862069, 161.09195402298852, 161.89655172413794, 162.64367816091954, 163.44827586206895, 164.19540229885058]
sigmaBergstrom = [0.025684931506849314, 0.059931506849315065, 0.09417808219178081, 0.1626712328767123, 0.1797945205479452, 0.2568493150684931, 0.2868150684931507, 0.3553082191780822, 0.41952054794520544, 0.4708904109589041, 0.547945205479452, 0.6035958904109588, 0.6592465753424657, 0.7320205479452054, 0.8390410958904109, 0.9375, 1.0273972602739725, 1.1130136986301369, 1.2200342465753424, 1.3398972602739725, 1.4683219178082192, 1.5582191780821917, 1.6780821917808217, 1.7851027397260273, 1.9135273972602738, 2.033390410958904, 2.127568493150685, 2.281678082191781, 2.4529109589041096]
sigmaBergstromPoint = [0, 0.05565068493150685, 0.07705479452054795, 0.14126712328767121, 0.1583904109589041, 0.23972602739726026, 0.2696917808219178, 0.3467465753424657, 0.4066780821917808, 0.4537671232876712, 0.5308219178082192, 0.5821917808219178, 0.646404109589041, 0.7148972602739726, 0.821917808219178, 0.9160958904109588, 1.0059931506849316, 1.0873287671232876, 1.2029109589041096, 1.3142123287671232, 1.4426369863013697, 1.5368150684931505, 1.6481164383561644, 1.7508561643835616, 1.8835616438356164, 1.9991438356164382, 2.0976027397260273, 2.2517123287671232, 2.422945205479452]
sigmaBergstromErrorMin = np.subtract(sigmaBergstrom,sigmaBergstromPoint)
sigmaBergstromErrorMax = sigmaBergstromErrorMin
errorBergstrom = [sigmaBergstromErrorMin, sigmaBergstromErrorMax]
plt.errorbar(gammaBergstrom,sigmaBergstrom,yerr=errorBergstrom,fmt="o");
plt.xlabel(r"$E_\gamma$ [MeV]")
plt.ylabel(r"$\sigma$ [$\mu$b]")

initial = [9,1.5]
popt, cov = curve_fit(sigma, gammaBergstrom, sigmaBergstrom, initial, sigmaBergstromErrorMin)
print(popt)
plt.title("$S=%0.2f$ MeV, $b=%0.2f$ fm" %(popt[0],popt[1]), x=0.5, y=0.8)
plt.xlabel(r"$E_\gamma$ [MeV]")
plt.ylabel(r"$\sigma$")
plt.tight_layout()
Photonenergy = np.linspace(gammaBergstrom[0],gammaBergstrom[28],29)
plt.plot(Photonenergy,sigma(Photonenergy,popt[0],popt[1]))
#save_fig("fit")
plt.show()
