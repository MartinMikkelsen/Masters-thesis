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

    return 10000*charge2/4/np.pi*mu/mp**2*q**3/k*np.sin(theta)**2*s**2*F(s)**2

Egammas = [151.4, 153.7, 149.1]
Ss = [86.2, 45.4, 35]
bs = [3.8, 3.9, 4.0]

angles = np.linspace(0,np.pi,150)



O = []
for i in angles:
   O.append(diffcross(Egammas[2],Ss[1],bs[1],i))
fig = plt.figure(figsize=(9,5.5))
ax = plt.axes(polar=False)
ax.plot(np.rad2deg(angles), O,label=r'$E_\gamma=%0.1f$ MeV' %(Egammas[2]),color='g')
AngleBeckD = np.array([12.429057888762769, 20.601589103291712, 28.43359818388195, 36.43586833144154, 44.43813847900113, 52.61066969353008, 60.44267877412032, 68.27468785471055, 76.4472190692395, 84.4494892167991, 92.45175936435868, 100.45402951191828, 108.45629965947786, 116.2883087400681, 124.29057888762769, 132.2928490351873, 140.12485811577753, 148.1271282633371, 156.1293984108967, 164.30192962542566, 172.64472190692393])
DiffCrossBeckD = np.array([0.02443064182194617, 0.00679089026915114, 0.017556935817805385, 0.015569358178053832, 0.024679089026915117, 0.010683229813664596, 0.027329192546583853, 0.02915113871635611, 0.04380952380952381, 0.03428571428571429, 0.04571428571428572, 0.04008281573498965, 0.030227743271221536, 0.04339544513457557, 0.04207039337474121, 0.03643892339544514, 0.0339544513457557, 0.04132505175983437, 0.028571428571428574, 0.031138716356107665, 0.009772256728778468])
ErrD = [0.01184265010351967, 0.0014078674948240168, 0.010351966873706006, 0.009689440993788821, 0.018053830227743272, 0.006459627329192547, 0.0212008281573499, 0.022939958592132506, 0.03635610766045549, 0.027743271221532095, 0.038178053830227744, 0.03279503105590063, 0.02360248447204969, 0.03494824016563147, 0.03296066252587992, 0.027246376811594204, 0.023850931677018634, 0.028571428571428574, 0.016149068322981366, 0.015072463768115944, 0.02244306418219462]
ErrorminD = np.subtract(DiffCrossBeckD,ErrD)
ErrormaxD = ErrorminD
ErrorBeckD = [ErrorminD, ErrormaxD]
plt.errorbar(AngleBeckD,DiffCrossBeckD,yerr=ErrorBeckD,fmt="o",color="g");


M = []
for i in angles:
   M.append(diffcross(Egammas[0],Ss[1],bs[1],i))

ax.plot(np.rad2deg(angles), M,label=r'$E_{\gamma}=%0.1f$ MeV' %(Egammas[0]),color='b')
AngleBeck = np.array([12.542955326460481, 20.61855670103093, 28.694158075601376, 36.597938144329895, 44.50171821305842, 52.40549828178694, 60.48109965635739, 68.21305841924399, 76.11683848797252, 84.02061855670104, 92.09621993127148, 100.34364261168385, 108.07560137457045, 116.1512027491409, 124.22680412371135, 132.3024054982818, 148.28178694158075, 156.18556701030928, 164.26116838487974])
DiffCrossBeck = np.array([0.009547738693467337, 0.022738693467336684, 0.033668341708542715, 0.03618090452261307, 0.019472361809045227, 0.05188442211055277, 0.04082914572864322, 0.06733668341708543, 0.05841708542713568, 0.0742462311557789, 0.0678391959798995, 0.05653266331658292, 0.07763819095477388, 0.05326633165829146, 0.05515075376884423, 0.05276381909547739, 0.028517587939698497, 0.04396984924623116, 0.03869346733668342])
Err = np.array([0.0023869346733668344, 0.013316582914572866, 0.023994974874371862, 0.02726130653266332, 0.013442211055276383, 0.042713567839195984, 0.0335427135678392, 0.05804020100502513, 0.050125628140703524, 0.06482412060301508, 0.05879396984924624, 0.047989949748743724, 0.06733668341708543, 0.04396984924623116, 0.04459798994974875, 0.04133165829145729, 0.017713567839195983, 0.027889447236180906, 0.019723618090452262])
Errormin = np.subtract(DiffCrossBeck,Err)
Errormax = Errormin
ErrorBeck = [Errormin, Errormax]
plt.errorbar(AngleBeck,DiffCrossBeck,yerr=ErrorBeck,fmt="o",color='b');


N = []
for i in angles:
   N.append(diffcross(Egammas[1],Ss[1],bs[1],i))
ax.plot(np.rad2deg(angles), N,label=r'$E_\gamma=%0.1f$ MeV' %(Egammas[1]),color='r')
AngleBeckB = np.array([12.06140350877193, 20.175438596491226, 28.07017543859649, 35.96491228070175, 44.07894736842105, 51.973684210526315, 59.868421052631575, 68.2017543859649, 76.09649122807018, 83.7719298245614, 91.8859649122807, 100, 107.89473684210526, 115.78947368421052, 124.12280701754385, 132.01754385964912, 140.1315789473684, 147.80701754385964, 156.140350877193, 164.25438596491227])
DiffCrossBeckB = np.array([0.019025522041763342, 0.05359628770301624, 0.056380510440835266, 0.0642691415313225, 0.08445475638051043, 0.08329466357308585, 0.10208816705336426, 0.07378190255220418, 0.09466357308584687, 0.11693735498839908, 0.07424593967517401, 0.05916473317865429, 0.08607888631090488, 0.08468677494199536, 0.06194895591647332, 0.09466357308584687, 0.060556844547563805, 0.11670533642691415, 0.05568445475638051, 0.06960556844547564])
ErrB = np.array([0.006728538283062645, 0.0382830626450116, 0.042923433874709975, 0.05174013921113689, 0.07169373549883991, 0.07169373549883991, 0.09025522041763341, 0.06403712296983759, 0.08375870069605569, 0.10510440835266821, 0.0642691415313225, 0.05034802784222738, 0.07470997679814385, 0.07262180974477958, 0.050580046403712296, 0.07888631090487239, 0.04617169373549884, 0.09234338747099768, 0.035498839907192575, 0.041299303944315545])
ErrorminB = np.subtract(DiffCrossBeckB,ErrB)
ErrormaxB = ErrorminB
ErrorBeckB = [ErrorminB, ErrormaxB]
plt.errorbar(AngleBeckB,DiffCrossBeckB,yerr=ErrorBeckB,fmt="o",color='r');
plt.legend(loc="best",frameon=False)
plt.xlabel(r"$\theta_q$ [deg]");
plt.ylabel(r"$d\sigma$/d$\Omega_q$ [$\mu$b/sr]");
plt.tight_layout()
plt.title("$S=%s$ MeV, $b=%s$ fm" %(Ss[1],bs[1]), x=0.25, y=0.9)
save_fig("MultiDiffcross_nonrel_2")
plt.figure()
