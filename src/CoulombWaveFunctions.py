from mpmath import *
mp.pretty =  True
mp.dps = 15

y = [-1, -4, -2.3, 1.6, 1, 2, 12, -9, -1, 0, 1.5] # (-x^10-4*x^9-2.3*x^8+1.6*x^7+x^6+2*x^5+12*x^4-9*x^3-x^2+1.5 = 0)

a = (polyroots(y), 5)

def func(x):
    Q = -1*x**10-4*x**9-2.3*x**8+1.6*x**7+1*x**6+2*x**5+12*x**4-9*x**3-1*x**2+1.5
    return Q

#your solutions
print(func(-3.17386869917664204976404))
print(func(1.03398886646284382528904))
