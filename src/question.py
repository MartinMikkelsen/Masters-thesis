import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.integrate import solve_bvp
from scipy.special import spherical_jn
from scipy.optimize import curve_fit

#Constants
m = 135  #MeV
mn = 939  #MeV
mu = m*mn/(mn+m) #Reduced mass
M = m+mn
g = 2*mu
hbarc = 197.3 #MeV fm
alpha = 1/137

def sigma(Egamma,S,b):

    Eq = Egamma-m
    k = Egamma/hbarc
    q = np.sqrt(2*mu*Eq)/hbarc
    s = q+mn/M*k

    frontfactors = np.sqrt(2)*8*np.pi**3*alpha*(mu/mn)**(3/2)

    dsigmadOmega = frontfactors*np.sqrt(Eq/mn)*(q**2/k)*F(s,S,b)**2

    return dsigmadOmega

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
    r = np.logspace(start,stop,num=1000,base=np.exp(1))
    E = -2

    u = [0*r,0*r,E*r/r[-1]]
    res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)

    phi = np.array(res.y.T[0:1000,0])
    Integral = simpson(spherical_jn(1,s*r)*phi*r**3, r, dx=0.001)

    return Integral

plt.figure(figsize=(9,5.5))
gammaFuchs = np.array([145.29, 146.11, 146.99, 147.82, 148.97, 149.83, 150.86, 151.69, 152.53, 153.37])
sigmaFuchs = np.array([0.056, 0.112, 0.158, 0.202, 0.284, 0.390, 0.462, 0.589, 0.676, 0.801])
errorFuchsmin = np.array([0.009, 0.011, 0.009, 0.014, 0.016, 0.017, 0.019, 0.026, 0.024, 0.027])
errorFuchsmax = errorFuchsmin
errorFuchs = [errorFuchsmin, errorFuchsmax]

plt.errorbar(gammaFuchs,sigmaFuchs,yerr=errorFuchs,fmt="o");
plt.xlabel(r"$E_\gamma$ [GeV]")
plt.ylabel(r"$\sigma$ [mb]")

Photonenergy = np.linspace(gammaFuchs[0],gammaFuchs[9],1000)
plt.plot(Photonenergy,sigma(Photonenergy,10,2)*10e5)
plt.show()
initial = [15,2]

# popt, cov = curve_fit(sigma, gammaFuchs, sigmaFuchs, initial, errorFuchsmax)
# print(popt)
# plt.title("$S=%0.2f$ MeV, $b=%0.2f$ fm" %(popt[0],popt[1]), x=0.5, y=0.8)
# plt.xlabel(r"$E_\gamma$ [MeV]")
# plt.ylabel(r"$\sigma$")
# plt.tight_layout()
# Photonenergy = np.linspace(gammaFuchs[0],gammaFuchs[9],1000)
# plt.plot(Photonenergy,sigma(Photonenergy,popt[0],popt[1]))
# #save_fig("fit")
# plt.show()
