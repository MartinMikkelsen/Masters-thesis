import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import trapz
from scipy.integrate import quad
from scipy.optimize import root
from scipy.special import spherical_jn
from scipy.special import hyp1f1
from scipy.special import gamma
from scipy.integrate import solve_bvp
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


def f(u,x):
    return (u[1],-1*u[1])
y0 = [0,1]
xs = np.linspace(1,5,500)
us = odeint(f,y0,xs)
ys = us[:,0]
plt.plot(xs,ys)
