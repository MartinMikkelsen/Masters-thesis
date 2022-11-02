import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, factorial
from scipy.integrate import quad
import scipy as sp
import seaborn as sns
import os
from pylab import plt, mpl

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
gamma = -2*charge2*mu*alpha/(hbarc**2*k)

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
    return np.array(First*integral,dtype='complex_')

def C(l,eta):
    return 2**l*np.exp(-np.pi*eta/2)*(abs(gamma(l+1+1j*eta))/(factorial(2*l+1)))
#Compare to mpmath

xes = [149.69199178644763, 152.36139630390144, 155.03080082135523, 157.5770020533881, 158.31622176591375, 160.32854209445586, 162.4229979466119, 162.99794661190964, 168.2546201232033,175.078125, 175.8984375]
RegularCoulomb(1,-2,2)
