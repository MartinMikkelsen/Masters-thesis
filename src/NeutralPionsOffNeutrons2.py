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


    xMAID = [144.6694214876033, 146.5702479338843, 148.1818181818182, 149.71074380165288, 151.28099173553719, 152.8512396694215, 154.4214876033058, 155.86776859504133, 157.39669421487605, 158.96694214876032, 160.45454545454547, 162.02479338842974, 163.51239669421489, 164.83471074380165, 166.28099173553719, 167.76859504132233, 169.17355371900825, 170.49586776859505, 171.8595041322314, 173.22314049586777, 174.58677685950414, 175.9090909090909, 177.1900826446281, 178.4297520661157, 179.7520661157025]
    yMAID = [-0.01440922190201729, 0.21613832853025935, 0.37463976945244953, 0.5187319884726225, 0.6772334293948127, 0.8357348703170029, 1.0230547550432276, 1.2103746397694524, 1.397694524495677, 1.6138328530259365, 1.8299711815561959, 2.0605187319884726, 2.319884726224784, 2.5504322766570606, 2.780979827089337, 3.0979827089337175, 3.357348703170029, 3.6455331412103744, 3.9481268011527377, 4.265129682997118, 4.596541786743516, 4.927953890489913, 5.259365994236311, 5.605187319884726, 5.965417867435158]

    plt.xlabel(r"$E_\gamma$ [MeV]");
    plt.ylabel(r"$\sigma [\mu b]$");
    initial = [50,3.5]
    popt, pcov = curve_fit(totalcross_rel,xMAID,yMAID)
    print("popt=",popt)
    print("Error=",np.sqrt(np.diag(pcov)))

    photonenergies1 = np.linspace(144.7,180,50)

    #plt.plot(photonenergies1,totalcross_rel(photonenergies1,69.33526458,3.60628741),label=r'$S=%0.1f$ MeV, $b=%0.1f$ fm, rel' %(69.33526458,3.60628741),color='r')
    #plt.plot(photonenergies1,totalcross(photonenergies1,69.33526458,3.60628741),label=r'$S=%0.1f$ MeV, $b=%0.1f$ fm, non-rel' %(69.33526458,3.60628741),linestyle='dashed',color='r')
    #plt.plot(photonenergies1,totalcross_rel(photonenergies1,57.9783878,3.97276793),label=r'$S=%0.1f$ MeV, $b=%0.1f$ fm, rel' %(57.9783878,3.97276793),color='g')
    #plt.plot(photonenergies1,totalcross(photonenergies1,57.9783878,3.97276793),label=r'$S=%0.1f$ MeV, $b=%0.1f$ fm, non-rel' %(57.9783878,3.97276793),linestyle='dashed',color='g')

    plt.plot(photonenergies1,totalcross_rel(photonenergies1,popt[0],popt[1]),label=r'$S=%0.1f$ MeV, $b=%0.1f$ fm' %(popt[0],popt[1]),color='r')
    plt.scatter(xMAID,yMAID)
    plt.legend(loc='best',frameon=False)
    #save_fig("ChargedPionOffProtonExact")
    plt.show()
