import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import trapz
from scipy.integrate import quad
from scipy.optimize import root
from scipy.special import spherical_jn
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

b = 1     #fm
S = 10    #MeV
m = 139.570   #MeV
mn = 938.2  #MeV
mu = m*mn/(mn+m) #Reduced mass
g = 2*mu

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

intphi = np.trapz(res.y.T[:,0], res.x,dx=0.001)
V = 1
N = 1/np.sqrt(V)*1/(np.sqrt(1+intphi))
alpha = 1/(137)
gamma = np.linspace(m,1600,np.size(res.x))
q = np.sqrt(2*mu*(gamma-m))
phi = res.y.T[:,0]
def Q(q):
    B = abs(np.trapz(spherical_jn(0,q-m*r)*r**4*phi,res.x,dx=0.001))**2
    return B

M = []
for i in q:
    M.append(Q(i))

omega = q**2/(2*mu)+m
D = 16/(9)*np.pi*N**2*alpha*(mu/m)**2

dsigmadomega = D*mu*q*omega*M
plt.figure(figsize=(9,5.5));

plt.plot(gamma/1000,dsigmadomega);
plt.plot(gamma/1000,1/5*dsigmadomega);
plt.legend(r"tes")


xes = [0.18800    , 0.20500   , 0.22300   , 0.24200   , 0.26000   , 0.26500   , 0.27900   , 0.29000   , 0.29800   , 0.31500   , 0.31800   , 0.33700   , 0.34000   , 0.35700   , 0.36500   , 0.37700   , 0.39000   , 0.39700   , 0.41500   , 0.41800   , 0.43800   , 0.44000   , 0.45900   , 0.46500   , 0.48100   , 0.49000   , 0.50200   , 0.51500   , 0.52400   , 0.54000   , 0.54600   , 0.56500   , 0.56800   , 0.59000   , 0.59000   , 0.61300   , 0.61500   , 0.63600   , 0.64000   , 0.65900   , 0.66100   , 0.66500   , 0.68200   , 0.69000   , 0.70600   , 0.71500   , 0.73000   , 0.74000   , 0.75400   , 0.76500   , 0.77800   , 0.79000   , 0.80000   , 0.80300   , 0.81500   , 0.82800   , 0.84000   , 0.85300   , 0.86500   , 0.87800   , 0.89000   , 0.90400   , 0.91500   , 0.92400   , 0.92900   , 0.94000   , 0.95600   , 0.96500   , 0.98200   , 0.99000   , 1.00800   , 1.01500   , 1.03500   , 1.04000   , 1.06200   , 1.06500   , 1.08900   , 1.09000   , 1.11500   , 1.11700   , 1.14000   , 1.14400   , 1.15000   , 1.16500   , 1.17200   , 1.19000   , 1.20000   , 1.21500   , 1.22900   , 1.24000   , 1.25800   , 1.26500   , 1.28600   , 1.29000   , 1.31500   , 1.31500   , 1.31600   , 1.34000   , 1.34500   , 1.36500   , 1.37500   , 1.39000   , 1.40400   , 1.41500   , 1.43500   , 1.44000   , 1.45000   , 1.46500   , 1.46500   , 1.49000   , 1.49500   , 1.50000   , 1.51500   , 1.52600   , 1.54000   , 1.55700   , 1.56500   , 1.58900   , 1.59000   , 1.61500]
yes = [7.8800E-02 , 0.11890   , 0.16820   , 0.20240   , 0.32340   , 0.42450   , 0.38710   , 0.48700   , 0.50420   , 0.52690   , 0.53260   , 0.54230   , 0.47780   , 0.48080   , 0.40660   , 0.41100   , 0.33410   , 0.31190   , 0.24440   , 0.24960   , 0.21090   , 0.22450   , 0.17420   , 0.20050   , 0.18880   , 0.17830   , 0.17640   , 0.17690   , 0.16710   , 0.18690   , 0.19120   , 0.19400   , 0.21850   , 0.21170   , 0.20910   , 0.23350   , 0.22260   , 0.23850   , 0.23270   , 0.24560   , 0.21100   , 0.23950   , 0.25140   , 0.26460   , 0.27390   , 0.27900   , 0.28910   , 0.27560   , 0.28660   , 0.26070   , 0.27510   , 0.24470   , 0.20100   , 0.25870   , 0.22110   , 0.23160   , 0.20630   , 0.22990   , 0.21370   , 0.22660   , 0.20880   , 0.19980   , 0.20170   , 0.19300   , 0.23360   , 0.20500   , 0.21170   , 0.20150   , 0.22100   , 0.21210   , 0.23240   , 0.21750   , 0.22340   , 0.21520   , 0.24160   , 0.19170   , 0.23240   , 0.19110   , 0.17470   , 0.22380   , 0.16500   , 0.20790   , 0.18210   , 0.15870   , 0.15930   , 0.16220   , 0.18410   , 0.14960   , 0.17530   , 0.14860   , 0.15260   , 0.14380   , 0.15560   , 0.15560   , 0.15000   , 0.14680   , 0.18060   , 0.15360   , 0.15990   , 0.15440   , 0.16430   , 0.15360   , 0.11600   , 0.14680   , 0.19680   , 0.15350   , 0.15370   , 0.17080   , 0.14420   , 0.15150   , 0.15810   , 0.15100   , 0.15590   , 0.17540   , 0.15410   , 0.16500   , 0.14610   , 0.16370   , 0.13880   , 0.15650]
y_errormin = [4.1E-02 , 3.8E-02, 3.4E-02, 3.1E-02, 3.2E-02, 8.0E-03, 3.4E-02, 8.1E-03, 3.7E-02, 8.1E-03, 3.7E-02, 3.2E-02, 8.3E-03, 3.0E-02, 8.0E-03, 3.1E-02, 7.6E-03, 3.3E-02, 7.5E-03, 2.6E-02, 2.6E-02, 6.7E-03, 2.7E-02, 6.6E-03, 2.6E-02, 6.5E-03, 2.5E-02, 6.4E-03, 1.7E-02, 6.2E-03, 1.7E-02, 4.0E-03, 1.7E-02, 3.9E-03, 1.8E-02, 1.6E-02, 4.1E-03, 1.6E-02, 4.2E-03, 1.8E-02, 2.2E-02, 4.3E-03, 1.8E-02, 4.5E-03, 1.9E-02, 4.5E-03, 2.0E-02, 4.6E-03, 1.9E-02, 4.6E-03, 1.8E-02, 4.6E-03, 2.0E-02, 1.6E-02, 4.6E-03, 1.7E-02, 4.6E-03, 1.7E-02, 4.5E-03, 1.6E-02, 4.5E-03, 1.7E-02, 4.7E-03, 1.9E-02, 1.8E-02, 4.6E-03, 1.6E-02, 4.8E-03, 1.5E-02, 4.9E-03, 1.5E-02, 5.0E-03, 1.6E-02, 4.9E-03, 1.7E-02, 5.0E-03, 1.8E-02, 5.0E-03, 5.0E-03, 1.6E-02, 4.8E-03, 1.5E-02, 7.8E-03, 4.9E-03, 1.7E-02, 5.0E-03, 1.8E-02, 4.8E-03, 1.5E-02, 5.0E-03, 1.5E-02, 4.9E-03, 1.6E-02, 4.7E-03, 1.4E-02, 4.9E-03, 1.6E-02, 4.8E-03, 1.6E-02, 5.0E-03, 1.7E-02, 5.4E-03, 1.7E-02, 5.2E-03, 1.6E-02, 5.3E-03, 6.3E-03, 1.4E-02, 5.4E-03, 5.2E-03, 1.4E-02, 9.0E-03, 5.3E-03, 1.6E-02, 5.1E-03, 1.6E-02, 5.1E-03, 1.5E-02, 5.1E-03, 5.1E-03]
y_errormax = y_errormin
y_error = [y_errormin,y_errormax]

plt.scatter(xes,yes);
plt.errorbar(xes,yes,yerr=y_error,fmt="o");
plt.xlabel(r"$E_\gamma$ [GeV]")
plt.ylabel(r"$\sigma_T^{\gamma p }$ [mb]")
plt.legend(r"$n\pi^+$ $p\pi^0$".split(),loc=0);
save_fig("comparingtheoryexperiment");