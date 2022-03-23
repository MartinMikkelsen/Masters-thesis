import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import root
from scipy.special import spherical_jn
import os
import seaborn as sns
from pylab import plt, mpl

sns.set_style("dark")
sns.set(font_scale=1)
sns.set_style("ticks")


b = 1     #fm
S = 10    #MeV
m = 139.570   #MeV
mn = 939.5  #MeV
mu = m*mn/(mn+m) #Reduced mass
hbar = 197.326
g = (2*mu)/hbar**2

factor = 100#(m/hbar)**(5)
print(g)

def f(r): #form factor
    return S*np.exp(-r**2/b**2)

def diff(phi,r,E):
    return (phi[1],-g*(E-m)*phi[0]-2/r*phi[1]+f(r)*g)

phi0 = [0,0] #Initial

def phi_fun(E):
    rs = np.linspace(1e-5,100,10000)
    ys = odeint(lambda phi,r: diff(phi,r,E), phi0, rs)
    integral = factor*12*np.pi*trapz(ys[:,0]*f(rs)*rs**4,rs)
    return integral - E

E_true = root(phi_fun, 20).x

rs = np.linspace(1e-5,100,10000)
ys = odeint(lambda phi,r: diff(phi,r,E_true), phi0, rs)

phi_true = ys[:,0]

plt.plot(rs, phi_true,linewidth=2,label=r'$\phi(x)$')
print("Minimum found at E =",E_true)

plt.title("Numerical solution",fontsize=14)
plt.xlabel("r [fm]",fontsize=14)
#plt.xticks([0,2,4,6,8,10],fontsize=14)
#plt.yticks([-0.2,0,0.2,0.4,0.6,0.8,1],fontsize=14)
plt.legend(loc=0,fontsize=14)
plt.figure(figsize=(10,6))
plt.savefig("phi.png")
plt.show()
