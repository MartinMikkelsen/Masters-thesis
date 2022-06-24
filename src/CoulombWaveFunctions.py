from mpmath import *
mp.pretty =  True
mp.dps = 15

F1 = lambda x: mp.coulombf(0,0,x)
F2 = lambda x: mp.coulombf(0,1,x)
F3 = lambda x: mp.coulombf(0,5,x)
F4 = lambda x: mp.coulombf(0,10,x)
F5 = lambda x: mp.coulombf(0,x/2,x)
plot([F1], [0,25], [-1.2,1.6])
