import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib as plt
import scipy.integrate as integrate
from scipy.special import spherical_jn
from scipy.integrate import solve_bvp
from scipy.integrate import trapz
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

infile = open(data_path("pgamma.csv"),'r')


b = 1     #fm
S = 10    #MeV
m = 139.570   #MeV
mn = 938.2  #MeV
mu = m*mn/(mn+m) #Reduced mass
g = (2*mu)

def f(r): #form factor
    return S*np.exp(-r**2/b**2)

def sys(r,u,E):
    y,v,I = u
    dy = v
    dv = g*(-E+m)*y-2/r*v+g*f(r)
    dI = f(r)*r**4*y
    return dy,dv,dI

def bc(ua, ub,E):
    ya,va,Ia = ua
    yb,vb,Ib = ub
    return va, vb+(g*(m+abs(E)))**0.5*yb, Ia, Ib-E

r = np.logspace(-5,0,5000)*5
E = -2

u = [0*r,0*r,E*r/r[-1]]
res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-5)

def plots():
    fig, ax = plt.subplots()
    plt.plot(res.x,res.y.T,'-',linewidth=3.5);
    plt.title("Numerical solution",size=15)
    plt.legend(r"$\phi$ $\phi'$ $I$".split(),loc=0);
    plt.xlabel("r [fm]")
    plt.show()

V = 1
Integral = 12*np.pi*V*np.trapz(res.y.T[:,0]**2*res.x**4)

psi0 = 1*np.sqrt(V)*1/(np.sqrt(1+Integral))

def d(k):
    return spherical_jn(0,k*res.x)*res.x**4*res.y.T[:,0]

df = pd.read_fwf(infile,usecols=(1,4,5,6), names=['PLAB(GEV/C)','SIG(MB)','STA_ERR+','STA_ERR-'])

xes = [0.18800,0.20500,0.22300,0.24200,0.26000,0.26500,0.27900,0.29000,0.29800,0.31500,0.31800,0.33700,0.34000,0.35700,0.36500,0.37700,0.39000,0.39700,0.41500,0.41800,0.43800,0.44000,0.45900,0.46500,0.48100,0.49000,0.50200,0.51500,0.52400,0.54000,0.54600,0.56500,0.56800,0.59000,0.59000,0.61300,0.61500,0.63600,0.64000,0.65900]
yes = [7.8800E-02,0.11890,0.16820,0.20240,0.32340,0.42450,0.38710,0.48700,0.50420,0.52690,0.53260,0.54230,0.47780,0.48080,0.40660,0.41100,0.33410,0.31190,0.24440,0.24960,0.21090,0.22450,0.17420,0.20050,0.18880,0.17830,0.17640,0.17690,0.16710,0.18690,0.19120,0.19400,0.21850,0.21170,0.20910,0.23350,0.22260,0.23850,0.23270,0.24560]
y_errormin = [4.1E-02,3.8E-02,3.4E-02,3.1E-02,3.2E-02,8.0E-03,3.4E-02,8.1E-03,3.7E-02,8.1E-03,3.7E-02,3.2E-02,8.3E-03,3.0E-02,8.0E-03,3.1E-02,7.6E-03,3.3E-02,7.5E-03,2.6E-02,2.6E-02,6.7E-03,2.7E-02,6.6E-03,2.6E-02,6.5E-03,2.5E-02,6.4E-03,1.7E-02,6.2E-03,1.7E-02,4.0E-03,1.7E-02,3.9E-03,1.8E-02,1.6E-02,4.1E-03,1.6E-02,4.2E-03,1.8E-02]
y_errormax = y_errormin
y_error = [y_errormin, y_errormax]

plt.scatter(xes,yes);
plt.errorbar(xes,yes,yerr=y_error,fmt="o");
plt.show()
gamma = np.linspace(m,140,np.size(res.x))
q = np.sqrt(2*mu*(gamma-m))

def normsquarematrixelement(k):
    Q = abs(trapz(spherical_jn(0,k*res.x)*res.y.T[:,0]*res.x**4))**2
    return Q

M2 = []
for i in q:
    M2.append(normsquarematrixelement(i))

V = 1
Integral = 12*np.pi*V*np.trapz(res.y.T[:,0]**2*res.x**4)

psi0 = 1*np.sqrt(V)*1/(np.sqrt(1+Integral))

d = 16*np.pi*psi0**2*mu**2/(137*9*m**2)*140*q*mu/(200**3)*3*(10*(10))**3*M2
plt.plot(q,d)

plt.show()
