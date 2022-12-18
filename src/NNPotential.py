import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import solve_bvp
from scipy.special import spherical_jn
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from tqdm import tqdm
from lmfit import Model
from scipy.special import gamma, factorial
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



rs = np.linspace(0.5,2.5,1000)
mu = 0.7
V = -10.463*np.exp(-mu*rs)/(mu*rs)-1650.6*np.exp(-4*mu*rs)/(mu*rs)+6484.2*np.exp(-7*mu*rs)/(mu*rs)
plt.figure(figsize=(9,5.5))
plt.plot(rs,V,linewidth=2.5)
plt.axhline(y=0,linestyle='dashed',color='r')
plt.ylim([-150,150])
plt.ylabel(r"$E$ [MeV]");
plt.xlabel(r"$r$ [fm]");
plt.title(r"$\pi, \rho, \omega, \sigma$", x=0.4, y=0.52);
plt.text(2.25,10,r"$\pi$",)
save_fig("NNPotential")
