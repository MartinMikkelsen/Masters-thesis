import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn
from scipy.integrate import solve_bvp
from scipy.special import gamma, factorial
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.integrate import quad
from numpy import angle
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
insidegamma = np.array(1+1+1.j*gamma, dtype=np.complex_)
sigmal = angle(sp.special.gamma(insidegamma))

C = 2*np.exp(-np.pi*gamma/2)*abs(sp.special.gamma(1+1+1.j*gamma))/(sp.special.factorial(2+1))

def F(eta,rho):
    F = 0
    for k in range(10):
        result += k


plt.plot(gamma,C*k**2)
