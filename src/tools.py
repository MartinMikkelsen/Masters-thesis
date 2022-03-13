import numpy as np
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum import InnerProduct
from sympy.physics.quantum.state import Ket, Bra
pi1 = smp.symbols('pi1')
pi2 = smp.symbols('pi2')
pi3 = smp.symbols('pi3')
x = smp.symbols('x', real=True)
y = smp.symbols('y', real=True)
z = smp.symbols('z', real=True)

a = smp.Matrix([[pi3,pi1-1j*pi2],[pi1+1j*pi2, -pi3]])
b = smp.Matrix([[z,x-1j*y],[x+1j*y, -z]])

Dagger(a)*a
