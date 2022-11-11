import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import root
from scipy.special import spherical_jn
from scipy.integrate import solve_bvp
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import seaborn as sns
import os
from pylab import plt, mpl
from tqdm import tqdm

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

m = 135.57  #MeV
mn = 939.272  #MeV
mu = m*mn/(mn+m) #Reduced mass
M = m+mn
g = (2*mu)
hbarc = 197.3 #MeV fm

def relativistic(S,b):
    def f(r): #form factor
        return S/b*np.exp(-r**2/b**2)

    def sys(r,u,E):
        y,v,z,l,I = u
        dy = v
        dv = z
        dz = l
        dl = 8*mu**3*(E-m)*y/(hbarc**4)-f(r)*8*mu**3/(hbarc**4)+4*mu**2*z/(hbarc**2)+(16*mu**2)*v/(r*(hbarc**2))-(6*l)/(r)
        dI = 12*np.pi*f(r)*r**4*y
        return dy,dv,dz,dl,dI

    def bc(ua, ub,E):
        ya,va,za,la,Ia = ua
        yb,vb,zb,lb,Ib = ub
        return va, vb,la,la-8/hbarc**4*mu**3*(E-m)*yb-4*mu**2/hbarc**2*vb, Ia, Ib-E,

    rmax = 5*b
    rmin = 0.01*b
    base1 = np.exp(1)
    start = np.log(rmin)
    stop = np.log(rmax)
    r = np.logspace(start,stop,num=3000,base=np.exp(1))
    E = -2

    u = [0*r,0*r,0*r,0*r,E*r/r[-1]]

    res2 = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)
    #print(res2.message,"The relativistic energy is: ",res2.p[0])
    return res2.x, res2.y.T[:,0], res2.y.T[:,1],res2.y.T[:,2],res2.p[0]

def nonrelativistic(S,b):
    def f(r): #form factor
        return S/b*np.exp(-r**2/b**2)

    def sys(r,u,E):
        y,v,I = u
        dy = v
        dv = g/(hbarc**2)*(-E+m)*y-4/r*v+g/(hbarc**2)*f(r)
        dI = 12*np.pi*f(r)*r**4*y
        return dy,dv,dI

    def bc(ua, ub,E):
        ya,va,Ia = ua
        yb,vb,Ib = ub
        return va, vb+(g*(m+abs(E)))**0.5*yb, Ia, Ib-E

    rmax = 5*b
    rmin = 0.01*b
    base1 = np.exp(1)
    start = np.log(rmin)
    stop = np.log(rmax)
    r = np.logspace(start,stop,num=3000,base=np.exp(1))
    E = -2

    u = [0*r,0*r,E*r/r[-1]]
    res = solve_bvp(sys,bc,r,u,p=[E],tol=1e-7,max_nodes=100000)
    return res.x, res.y.T[:,0], res.y.T[:,1],res.y.T[:,2],res.p[0]

plt.figure(figsize=(9,5.5));
# [a1,a2,a3,a4,a5] = relativistic(41.5,3.9)
# [b1,b2,b3,b4,b5] = nonrelativistic(41.5,3.9)
# S_values = [15,30,45]
# b_values = [2.5,3.5,4.5]
# print('The energy ratio is:', a5/b5)
# plt.plot(relativistic(S_values[0],b_values[0])[0], -relativistic(S_values[0],b_values[0])[0]*relativistic(S_values[0],b_values[0])[1],linewidth=3.5,linestyle='dashed',label=r'relativistic, $S=$%0.1f MeV, $b=$%0.1f fm' %(S_values[0],b_values[0]),color='r')
# plt.plot(nonrelativistic(S_values[0],b_values[0])[0], -nonrelativistic(S_values[0],b_values[0])[0]*nonrelativistic(S_values[0],b_values[0])[1],linewidth=3.5,label=r'non-relativistic',color='r')
#
# plt.plot(relativistic(S_values[1],b_values[1])[0], -relativistic(S_values[1],b_values[1])[0]*relativistic(S_values[1],b_values[1])[1],linewidth=3.5,linestyle='dashed',label=r'relativistic, $S=$%0.1f MeV, $b=$%0.1f fm' %(S_values[1],b_values[1]),color='g')
# plt.plot(nonrelativistic(S_values[1],b_values[1])[0], -nonrelativistic(S_values[1],b_values[1])[0]*nonrelativistic(S_values[1],b_values[1])[1],linewidth=3.5,label=r'non-relativistic',color='g')
#
# plt.plot(relativistic(S_values[2],b_values[2])[0], -relativistic(S_values[2],b_values[2])[0]*relativistic(S_values[2],b_values[2])[1],linewidth=3.5,linestyle='dashed',label=r'relativistic, $S=$%0.1f MeV, $b=$%0.1f fm' %(S_values[2],b_values[2]),color='navy')
# plt.plot(nonrelativistic(S_values[2],b_values[2])[0], -nonrelativistic(S_values[2],b_values[2])[0]*nonrelativistic(S_values[2],b_values[2])[1],linewidth=3.5,label=r'non-relativistic',color='navy')
# plt.legend(loc='best',frameon=False)
# plt.ylabel(r"$r\phi(r)$ [fm$^{-3/2}$]")
# plt.xlabel("r [fm]")
#save_fig("rela_vs_nonrela_radial")

#plt.plot(a1, a3,linewidth=3.5,linestyle='dashed', color='b')
#plt.plot(b1, b3,linewidth=3.5, color='b')



Ss = np.linspace(10,100,50)
bs = np.linspace(1,5,50)

S_energy1 = [relativistic(i,2.5)[4]/nonrelativistic(i,2.5)[4] for i in tqdm(Ss)]
b_energy1 = [relativistic(15,i)[4]/nonrelativistic(15,i)[4] for i in tqdm(bs)]
S_energy2 = [relativistic(i,3.5)[4]/nonrelativistic(i,3.5)[4] for i in tqdm(Ss)]
b_energy2 = [relativistic(30,i)[4]/nonrelativistic(30,i)[4] for i in tqdm(bs)]
S_energy3 = [relativistic(i,4.5)[4]/nonrelativistic(i,4.5)[4] for i in tqdm(Ss)]
b_energy3 = [relativistic(45,i)[4]/nonrelativistic(45,i)[4] for i in tqdm(bs)]

plt.figure(figsize=(9,5.5));
plt.plot(bs,b_energy1,label=r'$S=15$',linewidth=2.5)
plt.plot(bs,b_energy2,label=r'$S=30$',linewidth=2.5)
plt.plot(bs,b_energy3,label=r'$S=45$',linewidth=2.5)
plt.legend(loc='best',frameon=False)
plt.hlines(1,1,5,linestyle='dashed',color='r',linewidth=2.5)
plt.xlabel(r'$b$ [fm]')
plt.ylabel(r'$E_R$')
#save_fig("bconvergenceYukawa")
plt.figure(figsize=(9,5.5));
plt.plot(Ss,S_energy1,label=r'$b=2.5$',linewidth=2.5)
plt.plot(Ss,S_energy2,label=r'$b=3.5$',linewidth=2.5)
plt.plot(Ss,S_energy3,label=r'$b=4.5$',linewidth=2.5)
plt.xlabel(r'$S$ [MeV]')
plt.ylabel(r'$E_R$')
plt.legend(loc='best',frameon=False)
plt.hlines(1,10,100,linestyle='dashed',color='r')
# # save_fig("SconvergenceYukawa")
