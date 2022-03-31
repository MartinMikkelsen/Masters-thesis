import numpy as np
import matplotlib.pyplot as plt
from idesolver import IDESolver


b=1     #fm
S=10    #MeV
m=135   #MeV
mn = 939.5  #MeV
mu = m*mn/(mn+m) #Reduced mass
g = 1/(2*mu)

def f(r): #form factor
    return S*np.exp(-r**2/b**2)

solver = IDESolver(
            x=np.linspace(1e-7, 100, 100),
            y_0=[1, 1/10],
            c=lambda x, y: [y[1],-2/x*y[1]-2*mu*(m+S*np.exp(-x**2/b**2))],
            d=lambda x: [0,-2*mu*12*np.pi],
            k = lambda x, y: [0,S*np.exp(-x**2/b**2)*x**4],
            f=lambda y: [0, y[0]],
            lower_bound=lambda x: 0,
            upper_bound=lambda x: np.infty,
)

solver.solve()
