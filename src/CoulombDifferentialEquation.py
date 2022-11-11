import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, factorial, spherical_jn
from scipy.integrate import quad
import scipy as sp
import seaborn as sns
import os
from pylab import plt, mpl
import mpmath as mp
mpl.rcParams['font.family'] = 'XCharter'
custom_params = {"axes.spines.right": True, "axes.spines.top": True}
sns.set_theme(style="ticks", rc=custom_params)
sns.set_context("talk")

PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)


def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".pdf", format='pdf',bbox_inches="tight")

b =  3.9    #fm
S = 45.5    #MeV
m = 135.57  #MeV
mn = 939.272  #MeV
mu = m*mn/(mn+m) #Reduced mass
M = m+mn
g = (2*mu)
hbarc = 197.3 #MeV fm
alpha = 1/137
charge2 = hbarc/(137)
Mpip = m+mn
Egamma = np.linspace(145,180,100)
Eq = Egamma-m-0.5*Egamma**2/(Mpip)
k = Egamma/hbarc

def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

def RegularCoulomb(l,eta,rho):
    First = rho**(l+1)*2**l*np.exp(1j*rho-(np.pi*eta/2))/(abs(gamma(l+1+1j*eta)))
    integral = complex_quadrature(lambda t: np.exp(-2*1j*rho*t)*t**(l+1j*eta)*(1-t)**(l-1j*eta),0,1)[0]
    return np.array(First*integral)

def C(l,eta):
    return 2**l*np.exp(-np.pi*eta/2)*(abs(gamma(l+1+1j*eta))/(factorial(2*l+1)))

plt.figure(figsize=(9,5.5));

xes = np.linspace(0,10,100)
funct1 = [RegularCoulomb(1,2,x) for x in xes]
funct2 = [RegularCoulomb(1,5,x) for x in xes]
funct3 = [RegularCoulomb(1,-2,x) for x in xes]
funct4 = [RegularCoulomb(1,-5,x) for x in xes]
plt.plot(xes,funct1,linewidth=2.5,label=r'$F_1(2,kr)$',color='r')
plt.plot(xes,funct2,linewidth=2.5,label=r'$F_1(5,kr)$',color='g')
plt.plot(xes,funct3,linewidth=2.5,label=r'$F_1(-2,kr)$',color='navy')
plt.plot(xes,funct4,linewidth=2.5,label=r'$F_1(-5,kr)$')
plt.legend(loc='best', frameon=False);
plt.xlabel(r'$kr$');
plt.ylim([-2.5,2.5]);
save_fig('AttRepulCoulomb')
plt.figure(figsize=(9,5.5));
funct5 = [RegularCoulomb(1,3,x) for x in xes]
funct6 = [RegularCoulomb(1,2.5,x) for x in xes]
funct7 = [RegularCoulomb(1,2,x) for x in xes]
funct8 = [RegularCoulomb(1,1.5,x) for x in xes]
funct9 = [RegularCoulomb(1,1,x) for x in xes]
funct10 = [RegularCoulomb(1,0.5,x) for x in xes]
funct11 = [RegularCoulomb(1,0,x) for x in xes]
plt.plot(xes,funct9,linewidth=2.5,label=r'$F_1(1,kr)$',color='navy')
plt.plot(xes,funct10,linewidth=2.5,label=r'$F_1(0.5,kr)$',color='g')
plt.plot(xes,funct11,linewidth=2.5,label=r'$F_1(0,kr)$',color='r')
plt.plot(xes,xes*spherical_jn(1,xes),linewidth=3.5,linestyle='dashed',color='k',label=r'$krj_1(kr)$')
plt.legend(loc='best', frameon=False);
save_fig('LimitBessel')
