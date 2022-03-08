import numpy as np
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid


R = smp.symbols('R', real=True)
k = smp.symbols('k')
r = smp.symbols('r')
x = smp.symbols('x', real=True)
S = smp.symbols('S', real=True)
b = smp.symbols('b', real=True)
#f1 = smp.besselj(1,k*r)*r**3*smp.exp(-k*r)/r
#smp.integrate(f1, (r,0,smp.oo))

f = lambda r: np.exp(-k*r)*r**2*sp.special.spherical_jn(1,k*r)
k = 10
sp.integrate.quad(f,2,smp.oo)
print(np.pi/3*1/137*)
