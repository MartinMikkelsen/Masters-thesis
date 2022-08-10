import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.integrate import quad
from scipy.integrate import solve_bvp
from scipy.special import spherical_jn
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from hankel import HankelTransform

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

    return CrossSection*10e6

def F(s,S,b):

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
    r = np.logspace(start,stop,num=22,base=np.exp(1))
    E = -2

    u = [0*r,0*r,E*r/r[-1]]
    res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)

    phi = res.y.T[:22,0]

    def F(s):
        integral = quad(Spline(r,phi*r**3*spherical_jn(1,s*r)),0,10)
        return integral[0]

    return F(s)

plt.figure(figsize=(9,5.5))

xdata = np.array([144.0358208955224, 145.07462686567163, 146.22089552238805, 147.40298507462686, 148.5134328358209, 149.69552238805971, 150.84179104477613, 151.95223880597015, 153.09850746268657, 154.2089552238806, 155.31940298507462, 156.53731343283582, 157.61194029850748, 158.79402985074626, 159.9044776119403, 161.01492537313433, 162.19701492537314, 163.30746268656716, 164.4179104477612, 165.6358208955224, 166.71044776119402, 167.82089552238807])
ydata = np.array([0.0398406374501992, 0.049800796812749, 0.11952191235059761, 0.2290836653386454, 0.3286852589641434, 0.448207171314741, 0.5677290836653386, 0.7171314741035857, 0.9760956175298805, 1.155378486055777, 1.3545816733067728, 1.593625498007968, 1.7729083665338645, 2.1115537848605577, 2.290836653386454, 2.6095617529880477, 2.958167330677291, 3.197211155378486, 3.585657370517928, 3.9840637450199203, 4.282868525896414, 4.711155378486056])
ydataPoint = np.array([0, 0, 0.059602649006622516, 0.1490066225165563, 0.24834437086092717, 0.3675496688741722, 0.4768211920529801, 0.6258278145695364, 0.8841059602649007, 1.0529801324503312, 1.2417218543046358, 1.4701986754966887, 1.6291390728476822, 1.947019867549669, 2.1357615894039736, 2.433774834437086, 2.76158940397351, 2.980132450331126, 3.3675496688741724, 3.7450331125827816, 4.052980132450331, 4.420529801324504])
errormin = np.subtract(ydata,ydataPoint)
errormax = errormin
sigmaerror = [errormin, errormax]
plt.errorbar(xdata,ydata,yerr=sigmaerror,fmt="o");
plt.xlabel(r"$E_\gamma$ [MeV]")
plt.ylabel(r"$\sigma$ [$\mu$b]")
initial = [10,4]

popt, cov = curve_fit(sigma, xdata, ydata, initial, errormax)
print(popt)
plt.title("$S=%0.2f$ MeV, $b=%0.2f$ fm" %(popt[0],popt[1]), x=0.5, y=0.8)
plt.xlabel(r"$E_\gamma$ [MeV]")
plt.ylabel(r"$\sigma$")
plt.tight_layout()
Photonenergy = np.linspace(xdata[0],xdata[21],22)
plt.plot(Photonenergy,sigma(Photonenergy,popt[0],popt[1]))
plt.show()
#save_fig("fit")
