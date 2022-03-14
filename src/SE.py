import numpy as np
import scipy as smp
import seaborn as sns
import matplotlib.pyplot as plt
from sympy import *
from sympy.physics.secondquant import *

S = 1
b = 1
r = np.linspace(1,100,100)
def f(r):
    f = S*np.exp((r*r)/(b**2))
    return f


V = 1
a = np.array([1,0])*1/np.sqrt(V)
np.transpose(a)

piminus = smp.symbols('pi-', real=True)
pi0 = smp.symbols('pi0', real=True)
piplus = smp.symbols('pi+', real=True)
x = smp.symbols('x', real=True)
y = smp.symbols('y', real=True)
z = smp.symbols('z', real=True)

tau1 = np.matrix('0 1; 1 0')
tau2 = np.matrix('0 -1j; 1j 0')
tau3 = np.matrix('1 0; 0 -1')
tau = np.array([tau1, tau2, tau3])
pions = np.array([piminus, pi0, piplus])

taudotpi = np.dot(tau1,piminus)+np.dot(tau2,pi0)+np.dot(tau3,piplus)
sigmadotr = np.dot(tau1,x)+np.dot(tau2,y)+np.dot(tau3,z)

W = (taudotpi)*(sigmadotr)
