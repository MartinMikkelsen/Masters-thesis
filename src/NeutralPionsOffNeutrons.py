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

m = 134.9768  #MeV
mp = 939.565378  #MeV
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
    s = np.sqrt(q**2+k**2*(mp/Mpip)**2+2*q*k*(mp/Mpip)*np.cos(theta))

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

    return np.array(10000*charge2/4/np.pi*mu/m**2*q**3/k*np.sin(theta)**2*s**2*F(s)**2)

def diffcross_rel(Egamma,S,b,theta):

    Eq = Egamma-m-0.5*Egamma**2/(Mpip)
    if Eq<0 : return 0
    k = Egamma/hbarc
    q = np.sqrt(2*mu*Eq)/(hbarc)
    s = np.sqrt(q**2+k**2*(mp/Mpip)**2+2*q*k*(mp/Mpip)*np.cos(theta))
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

    return 10000*charge2/8/np.pi*dp2dEq/m**2*q**3/k*np.sin(theta)**2*s**2*F(s)**2

def totalcross(Egamma,S,b):
    func = lambda theta: 2*np.pi*np.sin(theta)*diffcross(Egamma,S,b,theta)
    integ = quad(func,0,np.pi)[0]
    return integ

def totalcross_rel(Egamma,S,b):
    func = lambda theta: 2*np.pi*np.sin(theta)*diffcross_rel(Egamma,S,b,theta)
    integ = quad(func,0,np.pi)[0]
    return integ

plt.figure(figsize=(9,5.5));


# photonenergies = np.linspace(144.7,180,50)
# N = []
# N_rel = []
# M = []
# M_rel = []
# P = []
# P_rel = []
# for i in tqdm(photonenergies):
#     N_rel.append(totalcross_rel(i,86.2,3.8))
#     N.append(totalcross(i,86.2,3.8))
#     M_rel.append(totalcross_rel(i,45.5,3.9))
#     M.append(totalcross(i,45.5,3.9))
#     P_rel.append(totalcross_rel(i,35.4,4.0))
#     P.append(totalcross(i,35.4,4.0))
#
#
# plt.plot(photonenergies,N,color='r',linestyle='dashed')
# plt.plot(photonenergies,N_rel, label=r'$S=86.2$ MeV, $b=3.8$ fm, rel', color='r')
# plt.plot(photonenergies,M,color='g',linestyle='dashed')
# plt.plot(photonenergies,M_rel, label=r'$S=45.5$ MeV, $b=3.9$ fm, rel', color='g')
# plt.plot(photonenergies,P,color='navy',linestyle='dashed')
# plt.plot(photonenergies,P_rel, label=r'$S=35.4$ MeV, $b=4.0$ fm, rel', color='navy')


xSAID = [144.21487603305786, 144.54545454545453, 145.12396694214877, 145.82644628099175, 146.77685950413223, 147.4793388429752, 148.26446280991735, 149.0909090909091, 149.83471074380165, 150.5785123966942, 151.44628099173553, 152.39669421487602, 153.1818181818182, 154.13223140495867, 155.04132231404958, 156.03305785123968, 157.02479338842974, 157.89256198347107, 158.84297520661158, 159.91735537190084, 160.86776859504133, 161.8181818181818, 163.01652892561984, 164.62809917355372, 165.5785123966942, 166.900826446281, 168.05785123966942, 169.6694214876033, 170.9504132231405, 172.1900826446281, 174.46280991735537, 176.03305785123968, 177.39669421487605, 179.25619834710744, 180.08264462809916]
ySAID = [-0.01440922190201729, 0.01440922190201729, 0.17291066282420747, 0.2737752161383285, 0.37463976945244953, 0.4610951008645533, 0.5331412103746398, 0.6340057636887608, 0.7348703170028819, 0.8069164265129682, 0.9221902017291066, 1.037463976945245, 1.1671469740634006, 1.3256484149855907, 1.4697406340057637, 1.6570605187319885, 1.8587896253602305, 2.031700288184438, 2.23342939481268, 2.4639769452449567, 2.6657060518731988, 2.8962536023054755, 3.155619596541787, 3.5302593659942363, 3.7896253602305476, 4.1354466858789625, 4.46685878962536, 4.942363112391931, 5.31700288184438, 5.677233429394812, 6.469740634005763, 6.974063400576369, 7.507204610951009, 8.24207492795389, 8.559077809798271]

xMAID = [144.6694214876033, 146.5702479338843, 148.1818181818182, 149.71074380165288, 151.28099173553719, 152.8512396694215, 154.4214876033058, 155.86776859504133, 157.39669421487605, 158.96694214876032, 160.45454545454547, 162.02479338842974, 163.51239669421489, 164.83471074380165, 166.28099173553719, 167.76859504132233, 169.17355371900825, 170.49586776859505, 171.8595041322314, 173.22314049586777, 174.58677685950414, 175.9090909090909, 177.1900826446281, 178.4297520661157, 179.7520661157025]
yMAID = [-0.01440922190201729, 0.21613832853025935, 0.37463976945244953, 0.5187319884726225, 0.6772334293948127, 0.8357348703170029, 1.0230547550432276, 1.2103746397694524, 1.397694524495677, 1.6138328530259365, 1.8299711815561959, 2.0605187319884726, 2.319884726224784, 2.5504322766570606, 2.780979827089337, 3.0979827089337175, 3.357348703170029, 3.6455331412103744, 3.9481268011527377, 4.265129682997118, 4.596541786743516, 4.927953890489913, 5.259365994236311, 5.605187319884726, 5.965417867435158]

popt, pcov = curve_fit(totalcross_rel,xSAID,ySAID,bounds=(0, [200,20]))
print("popt=",popt)
print("Error=",np.sqrt(np.diag(pcov)))

plt.xlabel(r"$E_\gamma$ [MeV]");
plt.ylabel(r"$\sigma [\mu b]$");
plt.legend(loc='best',frameon=False)

#save_fig('NeutralPionsOffNeutrons')

plt.show()
