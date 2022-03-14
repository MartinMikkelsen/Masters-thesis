import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import root

b = 1     #fm
S = 10    #MeV
m = 135   #MeV
mn = 939.5  #MeV
mu = m*mn/(mn+m) #Reduced mass
g = 1/(2*mu)

def f(r): #form factor
    return S*np.exp(-r**2/b**2)

def diff(phi,r,E):
    return (phi[1],(E-m)*phi[0]-2/r*phi[1]-f(r))

phi0 = [0,0] #Initial

def phi_fun(E):
    rs = np.linspace(1e-5,10,1000)
    ys = odeint(lambda phi,r: diff(phi,r,E), phi0, rs)
    integral = 12*np.pi*trapz(ys[:,0]*f(rs)*rs**4,rs)
    return integral - E

E_true = root(phi_fun, -13).x

rs = np.linspace(1e-5,10,1000)
ys = odeint(lambda phi,r: diff(phi,r,E_true), phi0, rs)

phi_true = ys[:,0]

plt.plot(rs, phi_true)
print("Minimum found at E =",E_true)

plt.title("$\phi(r)$")
plt.xlabel("r [fm]")
plt.savefig("phi.png")
plt.show()
