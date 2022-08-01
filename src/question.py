import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from hankel import HankelTransform

#Constants
m = 135
mn = 939
mp = 938
mu = m*mn/(mn+m)
M = m+mn
g = 2*mu
hbarc = 197.3
alpha = 1/137

def sigma(Egamma,S,b):

    Eq = Egamma-m
    k = Egamma/hbarc
    q = np.sqrt(2*mu*Eq)/hbarc
    s = q+mp/M*k

    frontfactors = np.sqrt(2)*8*np.pi**3*alpha*(mu/m)**(3/2)

    CrossSection = frontfactors*np.sqrt(Eq/m)*(q**2/k)*F(s,S,b)**2

    return CrossSection

def F(s,S,b):

    #To get phi
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
    r = np.logspace(start,stop,num=1000,base=np.exp(1))
    E = -2

    u = [0*r,0*r,E*r/r[-1]]
    res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)

    phi = res.y.T[:1000,0]

    # j_l(z) = √\frac{π}{2z} J_{l+1/2}(z)

    func = Spline(r,phi**2*np.sqrt(np.pi/(2*s*r)))

    ht = HankelTransform(
        nu= 3/2,     # The order of the bessel function
        N = 150,     # Number of steps in the integration
        h = 0.05     # Proxy for "size" of steps in integration
    )
    Fs = ht.transform(func,s,ret_err=False)
    return Fs

xdata = np.array([145.29, 146.11, 146.99, 147.82, 148.97, 149.83, 150.86, 151.69, 152.53, 153.37])
ydata = np.array([0.056, 0.112, 0.158, 0.202, 0.284, 0.390, 0.462, 0.589, 0.676, 0.801])
errorlower = np.array([0.009, 0.011, 0.009, 0.014, 0.016, 0.017, 0.019, 0.026, 0.024, 0.027])
errorhigher = errorlower
errordata = [errorlower, errorhigher]
plt.errorbar(xdata,ydata,yerr=errordata,fmt="o");
plt.xlabel(r"$E_\gamma$ [MeV]")
plt.ylabel(r"$\sigma$ [$\mu$b]")

Photonenergy = np.linspace(xdata[0],xdata[9],1000)
plt.plot(Photonenergy,sigma(Photonenergy,20,2)*10e6)
