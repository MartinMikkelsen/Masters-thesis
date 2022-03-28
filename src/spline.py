import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import root
from scipy.special import spherical_jn
import seaborn as sns
from scipy.optimize import minimize
sns.set(font_scale=1)
sns.set_style("ticks")

b=1     #fm
S=20    #MeV
m=135   #MeV
mn = 0#939.5  #MeV
mu = m*mn/(mn+m) #Reduced mass

factor = (2*mu)

def f(r): #form factor
    return S*np.exp(-r**2/b**2)

def diff(phi,r,E):
    return (phi[1],factor*(-E+m)*phi[0]-2/r*phi[1]+factor*f(r))

phi0 = [b/m,b/m] #Initial

def phi_fun(E):
    rs = np.linspace(1e-5,50,1000)
    ys = odeint(lambda phi,r: diff(phi,r,E), phi0, rs)
    integral = 12*np.pi*trapz(ys[:,0]*f(rs)*rs**4,rs)
    return integral - E

Eintervals = np.linspace(0,500,100)

g = []
for i in Eintervals:
    g.append(phi_fun(i))

plt.plot(Eintervals,g,'--')
plt.show()
