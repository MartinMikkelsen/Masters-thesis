import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import root
from scipy.special import spherical_jn
from scipy.special import spherical_jn
from scipy.integrate import solve_bvp
from scipy import fft
from sympy import hankel_transform, inverse_hankel_transform
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.integrate import quad
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


hbarc = 197.3 #MeV fm
alpha = 1/137
Egamma =  np.linspace(0,20,200)
bind = 2.225
M = 1875.612928
sigma = 8*np.pi/3*alpha*(hbarc)**2/M*(bind)**(0.5)*(Egamma-bind)**(3/2)/(Egamma**3)
plt.figure(figsize=(9,5.5))
plt.plot(Egamma,sigma,linewidth=2.5)
plt.legend(frameon=False);
plt.xlabel(r"$E_\gamma$ [MeV]");
plt.ylabel(r"$\sigma$ [10 mb]");
save_fig("Deuteron")
