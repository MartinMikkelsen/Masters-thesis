import numpy as np
import matplotlib.pyplot as plt
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

#This file is the raw data extracted from different papers. The papers are
#PhysRevLett.65.1841.pdf
#Genzel1974_Article_PhotoproductionOfNeutralPionsO.pdf
#Neutral-pion-photoproduction-from-the-proton-near-thres_1996_Physics-Letters.pdf
#PhysRevLett.57.3144.pdf
#Photoproduction-of-neutral-pions-on-hydrogen-at-photon-ener_1970_Nuclear-Phy.pdf
#These only consider pion photoproduction from a proton.

#The structure is as follows: variableNameOfFirstAuthor

gammaFuchs = [145.29, 146.11, 146.99, 147.82, 148.97, 149.83, 150.86, 151.69, 152.53, 153.37]
sigmaFuchs = [0.056, 0.112, 0.158, 0.202, 0.284, 0.390, 0.462, 0.589, 0.676, 0.801]
errorFuchsmin = [0.009, 0.011, 0.009, 0.014, 0.016, 0.017, 0.019, 0.026, 0.024, 0.027]
errorFuchsmax = errorFuchsmin
errorFuchs = [errorFuchsmin, errorFuchsmax]

#plt.scatter(gammaFuchs,sigmaFuchs);
#plt.errorbar(gammaFuchs,sigmaFuchs,yerr=errorFuchs,fmt="o");
#plt.xlabel(r"$E_\gamma$ [GeV]")
#plt.ylabel(r"$\sigma$ [mb]")

gammaMazzucato = [146.5, 146.6, 147.5, 147.6, 148.5, 150.5, 152.4, 154.1, 154.6, 159.0, 167.1, 169.2]
sigmaMazzucato = [0.25, 0.30, 0.30, 0.20, 0.29, 0.52, 0.86, 1.34, 1.16, 2.44, 5.42, 6.01]
errorMazzucatomin = [0.12, 0.14, 0.09, 0.08, 0.07, 0.08, 0.12, 0.20, 0.16, 0.20, 0.34, 0.58]
errorMazzucatomax = errorMazzucatomin
errorMazzucato = [errorMazzucatomin, errorMazzucatomax]

#plt.scatter(gammaMazzucato,sigmaMazzucato);
#plt.errorbar(gammaMazzucato,sigmaMazzucato,yerr=errorMazzucato,fmt="o");
#plt.xlabel(r"$E_\gamma$ [GeV]")
#plt.ylabel(r"$\sigma$ [mb]")


angleFischer = [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160]
gammmaFischer = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440]
diffcrossFischer200 = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2.05,np.nan, 2.02,np.nan, 2.14,np.nan, 1.92, np.nan, np.nan, np.nan, np.nan]
diffcrossFischer200Errormin = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.17,np.nan, 0.16,np.nan, 0.13,np.nan, 0.12, np.nan, np.nan, np.nan, np.nan]
diffcrossFischer200Errormax = diffcrossFischer200Errormin
diffcrossFischer200Error = [diffcrossFischer200Errormin, diffcrossFischer200Errormax]

diffcrossFischer210 = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,2.95,np.nan,2.76,np.nan,2.83, np.nan, 2.81, np.nan, np.nan, np.nan, np.nan]
diffcrossFischer210Errormin = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,0.11,np.nan,0.09,np.nan,0.10, np.nan, 0.12, np.nan, np.nan, np.nan, np.nan]
diffcrossFischer210Errormax = diffcrossFischer210Errormin
diffcrossFischer210Error = [diffcrossFischer210Errormin, diffcrossFischer210Errormax]

diffcrossFischer220 = [np.nan,np.nan,np.nan,np.nan,3.97,np.nan,3.73,np.nan,3.98,np.nan,3.99,np.nan,3.95,np.nan,3.67,np.nan,3.42,np.nan,np.nan]
diffcrossFischer220Errormin = [np.nan,np.nan,np.nan,np.nan,0.22,np.nan,0.09,np.nan,0.11,np.nan,0.09,np.nan,0.10,np.nan,0.07,np.nan,0.11,np.nan,np.nan]
diffcrossFischer220Errormax = diffcrossFischer220Errormin
diffcrossFischer220Error = [diffcrossFischer220Errormin, diffcrossFischer220Errormax]

diffcrossFischer230 = [np.nan,np.nan,4.98,np.nan,5.16,np.nan,4.91,np.nan,5.50,np.nan,5.24,np.nan,5.01,np.nan,4.74,np.nan,np.nan,np.nan,np.nan]
diffcrossFischer230Errormin = [np.nan,np.nan,0.48,np.nan,0.58,np.nan,0.15,np.nan,0.20,np.nan,0.17,np.nan,0.16,np.nan,0.15,np.nan,np.nan,np.nan,np.nan]
diffcrossFischer230Errormax = diffcrossFischer230Errormin
diffcrossFischer230Error = [diffcrossFischer230Errormin, diffcrossFischer230Errormax]

diffcrossFischer240 = [np.nan,np.nan,6.68,np.nan,7.00,np.nan,7.32,np.nan,7.26,np.nan,6.71,np.nan,6.48,np.nan,5.84,np.nan,5.53,np.nan,np.nan]
diffcrossFischer240Errormin = [np.nan,np.nan,0.27,np.nan,0.18,np.nan,0.15,np.nan,0.14,np.nan,0.14,np.nan,0.14,np.nan,0.11,np.nan,0.16,np.nan,np.nan]
diffcrossFischer240Errormax = diffcrossFischer240Errormin
diffcrossFischer240Error = [diffcrossFischer240Errormin, diffcrossFischer240Errormax]


diffcrossFischer250 = [6.40,np.nan,8.70,np.nan,10.15,np.nan,9.09,np.nan,9.01,np.nan,8.72,np.nan,8.52,np.nan,7.72,np.nan,np.nan,np.nan,np.nan]
diffcrossFischer250Errormin = [0.52,np.nan,0.41,np.nan,0.29,np.nan,0.24,np.nan,0.22,np.nan,0.26,np.nan,0.19,np.nan,0.19,np.nan,np.nan,np.nan,np.nan]
diffcrossFischer250Errormax = diffcrossFischer250Errormin
diffcrossFischer250Error = [diffcrossFischer250Errormin, diffcrossFischer250Errormax]

diffcrossFischer260 = [9.47,11.57,11.86,12.24,13.05,12.81,12.62,12.59,12.33,11.61,11.35,11.11,10.48,10.16,9.44,8.56,8.06,7.89,7.44]
diffcrossFischer260Errormin = [0.54,0.49,0.19,0.50,0.22,0.16,0.15,0.15,0.15,0.12,0.11,0.17,0.13,0.13,0.11,0.12,0.13,0.12,0.10]
diffcrossFischer260Errormax = diffcrossFischer260Errormin
diffcrossFischer260Error = [diffcrossFischer260Errormin, diffcrossFischer260Errormax]

diffcrossFischer270 = [13.20,14.11,14.90,16.11,16.37,16.18,15.92,16.08,15.17,14.97,14.45,13.57,12.83,12.19,11.68,10.73,10.01,9.70,8.91]
diffcrossFischer270Errormin = [0.59,0.58,0.27,0.44,0.16,0.22,0.16,0.22,0.13,0.19,0.19,0.21,0.18,0.19,0.15,0.21,0.22,0.23,0.18]
diffcrossFischer270Errormax = diffcrossFischer270Errormin
diffcrossFischer270Error = [diffcrossFischer270Errormin, diffcrossFischer270Errormax]

diffcrossFischer280 = [17.36,18.56,19.25,19.48,20.11,20.24,19.99,19.10,19.28,18.31,17.33,16.69,15.88,14.83,13.48,12.43,11.63,11.37,10.64]
diffcrossFischer280Errormin = [0.40,0.44,0.30,0.51,0.18,0.26,0.21,0.23,0.19,0.27,0.17,0.25,0.15,0.18,0.16,0.17,0.16,0.13,0.15]
diffcrossFischer280Errormax = diffcrossFischer280Errormin
diffcrossFischer280Error = [diffcrossFischer280Errormin, diffcrossFischer280Errormax]

diffcrossFischer300 = [np.nan,np.nan,19.29,22.49,24.16,26.01,26.51,26.72,27.40,26.80,26.44,25.92,25.69,24.14,22.84,22.00,20.42,18.83,17.68,16.44,14.27,14.00,13.01]
diffcrossFischer300Errormin = [np.nan, np.nan,0.64,0.91,0.33,0.66,0.41,0.28,0.26,0.33,0.24,0.25,0.24,0.32,0.24,0.45,0.20,0.25,0.17,0.32,0.29,0.20,0.25]
diffcrossFischer300Errormax = diffcrossFischer300Errormin
diffcrossFischer300Error = [diffcrossFischer300Errormin, diffcrossFischer300Errormax]

angleFischer2 = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160]


plt.figure(figsize=(9,5.5))

plt.errorbar(angleFischer, diffcrossFischer230, yerr=diffcrossFischer230Error, fmt='o');
plt.errorbar(angleFischer, diffcrossFischer240, yerr=diffcrossFischer240Error, fmt='o');
plt.errorbar(angleFischer, diffcrossFischer260, yerr=diffcrossFischer260Error, fmt='o');
plt.errorbar(angleFischer2, diffcrossFischer300, yerr=diffcrossFischer300Error, fmt='o');
plt.legend(r"$E_\gamma=230\,MeV$ $E_\gamma=240\,MeV$ $E_\gamma=260\,MeV$ $E\gamma=300\,MeV$".split(),loc=0,frameon=False);
plt.xlabel(r"$\theta_{c.m}^\pi$");
plt.ylabel(r"$\frac{\mu b}{sr}$");
save_fig("Fischerdata")
