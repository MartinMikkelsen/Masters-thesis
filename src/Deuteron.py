import numpy as np
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="dark")

m = 469.4  # MeV
E = 1.11  # MeV
U0 = 24.82  # MeV
R = 2.127  # fm
hbar = 6.6e-36
k = 1.001
c = 1.001
A = 1
D = 7.1
r = np.linspace(0, 10, 100)  # fm
r1 = np.linspace(0, R, 100)
r2 = np.linspace(R, 10, 100)

u1 = A*np.sin(k*r1)
u2 = D*np.exp(-c*r2)
u = u1+u2
