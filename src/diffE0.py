import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp


b=1     #fm
S=10    #MeV
m=135   #MeV
mn = 939.5  #MeV
mu = m*mn/(mn+m) #Reduced mass

g = (2*mu)

def f(r): #form factor
    return S*np.exp(-r**2/b**2)

def diff(phi,r):
    return (phi[1],(+2.2+m)*phi[0]-2/r*phi[1]+f(r))


phi0 = [0.07,1] #Initial

r = np.linspace(1e-7,20,100)

sol = odeint(diff,phi0,r)

r_sol = sol.T[0]
phi_sol = sol.T[1]

#plt.plot(r,phi_sol)
#plt.show()
