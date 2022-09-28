import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import solve_bvp
from scipy.special import spherical_jn
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

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
    if Eq.any()<0 : return 0
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

    def F(S):
        func = lambda r: phi3(r)*r**3*spherical_jn(1,S*r)
        integral =  4*np.pi/s*quad(func,0,rmax)[0]
        return integral

    return 10000*charge2/4/np.pi*mu/mp**2*q**3/k*np.sin(theta)**2*s**2*F(s)**2 #10000 to match units

def totalcross(Egamma,S,b):
    func = lambda theta: 2*np.pi*np.sin(theta)*diffcross(Egamma,S,b,theta)
    integ = quad(func,0,np.pi)[0]
    return integ

plt.figure(figsize=(9,5.5));

xdata = np.array([144.0358208955224, 145.07462686567163, 146.22089552238805, 147.40298507462686, 148.5134328358209, 149.69552238805971, 150.84179104477613, 151.95223880597015, 153.09850746268657, 154.2089552238806, 155.31940298507462, 156.53731343283582, 157.61194029850748, 158.79402985074626, 159.9044776119403, 161.01492537313433, 162.19701492537314, 163.30746268656716, 164.4179104477612, 165.6358208955224, 166.71044776119402, 167.82089552238807])
ydata = np.array([0.0398406374501992, 0.049800796812749, 0.11952191235059761, 0.2290836653386454, 0.3286852589641434, 0.448207171314741, 0.5677290836653386, 0.7171314741035857, 0.9760956175298805, 1.155378486055777, 1.3545816733067728, 1.593625498007968, 1.7729083665338645, 2.1115537848605577, 2.290836653386454, 2.6095617529880477, 2.958167330677291, 3.197211155378486, 3.585657370517928, 3.9840637450199203, 4.282868525896414, 4.711155378486056])
yerr = np.array([0, 0, 0.059602649006622516, 0.1490066225165563, 0.24834437086092717, 0.3675496688741722, 0.4768211920529801, 0.6258278145695364, 0.8841059602649007, 1.0529801324503312, 1.2417218543046358, 1.4701986754966887, 1.6291390728476822, 1.947019867549669, 2.1357615894039736, 2.433774834437086, 2.76158940397351, 2.980132450331126, 3.3675496688741724, 3.7450331125827816, 4.052980132450331, 4.420529801324504])
errmin = np.subtract(ydata,yerr)
errmax = errmin
err = [errmin, errmax]
plt.errorbar(xdata,ydata,yerr=errmin,fmt="o",color='b')

plt.xlabel(r"$E_\gamma$ [MeV]");
plt.ylabel(r"$\sigma [\mu b]$");
plt.grid()

# photonenergies = np.linspace(144.7,170,50)
# N = []
#
# for i in (photonenergies):
#     N.append(totalcross(i,86.2,3.8))
#
# plt.plot(photonenergies,N, label=r'$S=86.2$ MeV, $b=3.8$ fm', color='r')

popt, pcov = curve_fit(totalcross, xdata, ydata)

plt.show()
