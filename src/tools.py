import numpy as np
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid

R = smp.symbols('R', real=True)
k = smp.symbols('k')
x = smp.symbols('x', real=True)
bessel1 = smp.besselj(1,x)
#f1 = smp.exp(-x)*x/(k*k)*bessel1
#smp.integrate(f1, x)

f = lambda x: np.exp(-x)*x*sp.special.spherical_jn(1,x)
sp.integrate.quad(f,0,10000)
