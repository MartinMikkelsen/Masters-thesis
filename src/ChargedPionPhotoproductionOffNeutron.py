import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.integrate import quad
from scipy.integrate import quadrature
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
from scipy.special import gamma, factorial

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

m = 139.57039  #MeV
mn = 939.565378  #MeV
mu = m*mn/(mn+m) #Reduced mass
g = 2*mu
hbarc = 197.327 #MeV fm
alpha = 1/137
charge2 = hbarc/(137)
Mpip = m+mn


def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

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
    res = solve_bvp(sys,bc,r2,u,p=[E],tol=1e-6,max_nodes=100000)

    phi = res.y.T[:np.size(r2),0]
    phi3 = Spline(r2,phi)


    def CoulombWave(l,eta,rho):
        First = rho**(l+1)*2**l*np.exp(1j*rho-(np.pi*eta/2),dtype='complex_')/(abs(gamma(l+1+1j*eta)))
        integral = complex_quadrature(lambda t: np.exp(-2*1j*rho*t,dtype='complex_')*t**(l+1j*eta)*(1-t)**(l-1j*eta),0,1)[0]
        return np.array(First*integral,dtype='complex_')

    def C(l,eta):
        return 2**l*np.exp(-np.pi*eta/2)*(abs(gamma(l+1+1j*eta))/(factorial(2*l+1)))

    def eta(S):
        return -charge2*mu/(hbarc**2*S)

    def F(S):
        func = lambda r: phi3(r)*r**3*CoulombWave(1,eta(S),S*r)
        integral =  4*np.pi/s*quad(func,0,rmax)[0]
        return integral

    return 10000*charge2/2/np.pi*mu/mn**2*q**3/k*np.sin(theta)**2*s**2*F(s)**2

def totalcross(x,S,b):
    tot = [quad(lambda theta: 2*np.pi*np.sin(theta)*diffcross(i,S,b,theta),0,np.pi)[0] for i in tqdm(x)]
    return tot

if __name__ == '__main__':

    plt.figure(figsize=(9,5.5));

    x = [149.69199178644763, 152.36139630390144, 155.03080082135523, 157.5770020533881, 158.31622176591375, 160.32854209445586, 162.4229979466119, 162.99794661190964]
    y = [31.942446043165468, 56.115107913669064, 71.22302158273381, 83.3093525179856, 80.28776978417265, 93.45323741007194, 100.79136690647482, 105.53956834532373]
    yprime = [22.230215827338128, 46.83453237410072, 61.72661870503597, 72.73381294964028, 75.75539568345323, 83.09352517985612, 89.35251798561151, 99.92805755395683]

    errorSchmidtmin = np.subtract(y,yprime)
    errorSchmidtmax = errorSchmidtmin
    sigmaErrorSchmidt = [errorSchmidtmin, errorSchmidtmax]
    plt.errorbar(x,y,yerr=sigmaErrorSchmidt,fmt="o");
    plt.xlabel(r"$E_\gamma$ [MeV]");
    plt.ylabel(r"$\sigma [\mu b]$");

    gmodel = Model(totalcross)
    result = gmodel.fit(y, x=x,S=45,b=2.5)
    print(result.fit_report())

    # x = [149.69199178644763, 152.36139630390144, 155.03080082135523, 157.5770020533881, 158.31622176591375, 160.32854209445586, 162.4229979466119, 162.99794661190964, 168.2546201232033,175.078125, 175.8984375]
    # y = [31.942446043165468, 56.115107913669064, 71.22302158273381, 83.3093525179856, 80.28776978417265, 93.45323741007194, 100.79136690647482, 105.53956834532373, 106.61870503597122,109.82142857142857, 130.8379120879121]
    # yprime = [22.230215827338128, 46.83453237410072, 61.72661870503597, 72.73381294964028, 75.75539568345323, 83.09352517985612, 89.35251798561151, 99.92805755395683, 101.0071942446043,89.01098901098901, 121.56593406593406]

    #photonenergies1 = np.linspace(148.4,163,25)
    #plt.plot(photonenergies1,totalcross(photonenergies1,23.7663192,10.0473523))
    #plt.legend(loc='best',frameon=False)
    #save_fig("")
    #plt.show()
