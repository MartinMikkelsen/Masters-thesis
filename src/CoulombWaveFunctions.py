from mpmath import *
import numpy as np
import matplotlib.pyplot as plt

mp.pretty =  True
mp.dps = 100

M = []
for x in range(0,25):
    M.append(coulombf(0, 1, x))

x = np.linspace(0,25,25)
