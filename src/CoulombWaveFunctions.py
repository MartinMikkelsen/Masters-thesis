import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import os
import scipy as sp
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from tqdm import tqdm
import mpmath
from pylab import plt, mpl
import seaborn as sns
from numpy import frompyfunc
from scipy.special import gamma, factorial
from scipy.special import spherical_jn
from sympy import hyper
from scipy.integrate import quad

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
mp = 939.565378  #MeV
mu = m*mp/(mp+m) #Reduced mass
g = 2*mu
hbarc = 197.327 #MeV fm
alpha = 1/137
charge2 = hbarc/(137)
Mpip = m+mp

def diffcross(Egamma,S,b,theta):

    Eq = Egamma-m-0.5*Egamma**2/(Mpip)
    #if Eq.any()<0 : return 0
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

    def eta(S):
        return -charge2*mu/(hbarc**2*S)

    def complex_quadrature(func, a, b, **kwargs):
        def real_func(x):
            return np.real(func(x))
        def imag_func(x):
            return np.imag(func(x))
        real_integral = quad(real_func, a, b, **kwargs)
        imag_integral = quad(imag_func, a, b, **kwargs)
        return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

    def RegularCoulomb(l,eta,rho):
        First = rho**(l+1)*2**l*np.exp(1j*rho-(np.pi*eta/2),dtype='complex_')/(abs(gamma(l+1+1j*eta)))
        integral = complex_quadrature(lambda t: np.exp(-2*1j*rho*t,dtype='complex_')*t**(l+1j*eta)*(1-t)**(l-1j*eta),0,1)[0]
        return np.array(First*integral,dtype='complex_')

    def C(l,eta):
        return 2**l*np.exp(-np.pi*eta/2)*(abs(gamma(l+1+1j*eta))/(factorial(2*l+1)))

    def IrregularCoulomb(l,eta,rho):
        First = np.exp(-1j*rho)*rho**(-l)/(factorial(2*l+1)*C(l,eta))
        integral = quad(lambda t: np.exp(-t)*t**(-l-1j*eta)*(t+2*1j*rho)**(l+1j*eta),0,np.infty)[0]
        return First*integral

    xes = np.linspace(0,10,100)
    coulombfwave = [RegularCoulomb(1,-2,i) for i in xes]

    F1 = lambda x: mpmath.coulombf(1,-2,x)
    mpmath.plot([F1], [0,10])
    plt.plot(xes,coulombfwave)
    return

diffcross(160,45,3.9,np.pi)
