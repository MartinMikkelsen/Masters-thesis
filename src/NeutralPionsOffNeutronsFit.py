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
from lmfit import Model
from multiprocessing import Pool
import time
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

m = 139.57039  #MeV
mn = 938.272088  #MeV
mu = m*mn/(mn+m) #Reduced mass
g = 2*mu
hbarc = 197.327 #MeV fm
alpha = 1/137
charge2 = hbarc/(137)
Mpip = m+mn


def diffcross(Egamma,S,b,theta):

    Eq = Egamma-m-0.5*Egamma**2/(Mpip)
    if Eq<0 : return 0
    k = Egamma/hbarc
    q = np.sqrt(2*mu*Eq)/(hbarc)
    s = np.sqrt(q**2+k**2*(m/Mpip)**2+2*q*k*(m/Mpip)*np.cos(theta))
    dp2dEq = ((Eq**2+2*Eq*mn+2*mn**2+2*Eq*m+2*mn*m)*(Eq**2+2*Eq*mn+2*m**2+2*Eq*m+2*mn*m))/(2*(Eq+mn+m)**3)

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
    res = solve_bvp(sys,bc,r2,u,p=[E],tol=1e-6,max_nodes=100000)

    phi = res.y.T[:np.size(r2),0]
    phi3 = Spline(r2,phi)

    def F(S):
        start = time.time()
        func = lambda r: phi3(r)*r**3*spherical_jn(1,S*r)
        integral =  4*np.pi/s*quad(func,0,rmax,limit=100)[0]
        #print(f"F took: {time.time()-start}")
        return integral

    return 10000*charge2/2/np.pi*mu/mn**2*q**3/k*np.sin(theta)**2*s**2*F(s)**2

def diffcross_rel(Egamma,S,b,theta):

    Eq = Egamma-m-0.5*Egamma**2/(Mpip)
    if Eq<0 : return 0
    k = Egamma/hbarc
    q = np.sqrt(2*mu*Eq)/(hbarc)
    s = np.sqrt(q**2+k**2*(m/Mpip)**2+2*q*k*(m/Mpip)*np.cos(theta))
    dp2dEq = ((Eq**2+2*Eq*mn+2*mn**2+2*Eq*m+2*mn*m)*(Eq**2+2*Eq*mn+2*m**2+2*Eq*m+2*mn*m))/(2*(Eq+mn+m)**3)

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
    res = solve_bvp(sys,bc,r2,u,p=[E],tol=1e-6,max_nodes=100000)

    phi = res.y.T[:np.size(r2),0]
    phi3 = Spline(r2,phi)

    def F(S):
        start = time.time()
        func = lambda r: phi3(r)*r**3*spherical_jn(1,S*r)
        integral =  4*np.pi/s*quad(func,0,rmax,limit=100)[0]
        #print(f"F took: {time.time()-start}")
        return integral

    return 10000*charge2/8/np.pi*dp2dEq/m**2*q**3/k*np.sin(theta)**2*s**2*F(s)**2

def totalcross(x,S,b):
    tot = [quad(lambda theta: 2*np.pi*np.sin(theta)*diffcross(i,S,b,theta),0,np.pi)[0] for i in tqdm(x)]
    return tot

def totalcross_rel(x,S,b):
    tot = [quad(lambda theta: 2*np.pi*np.sin(theta)*diffcross_rel(i,S,b,theta),0,np.pi)[0] for i in tqdm(x)]
    return tot

if __name__ == '__main__':

    plt.figure(figsize=(9,5.5));



    xSAID = [144.21487603305786, 144.54545454545453, 145.12396694214877, 145.82644628099175, 146.77685950413223, 147.4793388429752, 148.26446280991735, 149.0909090909091, 149.83471074380165, 150.5785123966942, 151.44628099173553, 152.39669421487602, 153.1818181818182, 154.13223140495867, 155.04132231404958, 156.03305785123968, 157.02479338842974, 157.89256198347107, 158.84297520661158, 159.91735537190084, 160.86776859504133, 161.8181818181818, 163.01652892561984, 164.62809917355372, 165.5785123966942, 166.900826446281, 168.05785123966942, 169.6694214876033, 170.9504132231405, 172.1900826446281, 174.46280991735537, 176.03305785123968, 177.39669421487605, 179.25619834710744, 180.08264462809916]
    ySAID = [-0.01440922190201729, 0.01440922190201729, 0.17291066282420747, 0.2737752161383285, 0.37463976945244953, 0.4610951008645533, 0.5331412103746398, 0.6340057636887608, 0.7348703170028819, 0.8069164265129682, 0.9221902017291066, 1.037463976945245, 1.1671469740634006, 1.3256484149855907, 1.4697406340057637, 1.6570605187319885, 1.8587896253602305, 2.031700288184438, 2.23342939481268, 2.4639769452449567, 2.6657060518731988, 2.8962536023054755, 3.155619596541787, 3.5302593659942363, 3.7896253602305476, 4.1354466858789625, 4.46685878962536, 4.942363112391931, 5.31700288184438, 5.677233429394812, 6.469740634005763, 6.974063400576369, 7.507204610951009, 8.24207492795389, 8.559077809798271]

    plt.xlabel(r"$E_\gamma$ [MeV]");
    plt.ylabel(r"$\sigma [\mu b]$");
    initial = [50,3.5]
    popt, pcov = curve_fit(totalcross_rel,xSAID,ySAID)
    print("popt=",popt)
    print("Error=",np.sqrt(np.diag(pcov)))

    photonenergies1 = np.linspace(144.7,180,50)

    #plt.plot(photonenergies1,totalcross_rel(photonenergies1,69.33526458,3.60628741),label=r'$S=%0.1f$ MeV, $b=%0.1f$ fm, rel' %(69.33526458,3.60628741),color='r')
    #plt.plot(photonenergies1,totalcross(photonenergies1,69.33526458,3.60628741),label=r'$S=%0.1f$ MeV, $b=%0.1f$ fm, non-rel' %(69.33526458,3.60628741),linestyle='dashed',color='r')
    #plt.plot(photonenergies1,totalcross_rel(photonenergies1,57.9783878,3.97276793),label=r'$S=%0.1f$ MeV, $b=%0.1f$ fm, rel' %(57.9783878,3.97276793),color='g')
    #plt.plot(photonenergies1,totalcross(photonenergies1,57.9783878,3.97276793),label=r'$S=%0.1f$ MeV, $b=%0.1f$ fm, non-rel' %(57.9783878,3.97276793),linestyle='dashed',color='g')

    plt.plot(photonenergies1,totalcross_rel(photonenergies1,popt[0],popt[1]),label=r'$S=%0.1f$ MeV, $b=%0.1f$ fm' %(popt[0],popt[1]),color='r')
    plt.scatter(xSAID,ySAID)
    plt.legend(loc='best',frameon=False)
    #save_fig("ChargedPionOffProtonExact")
    plt.show()
