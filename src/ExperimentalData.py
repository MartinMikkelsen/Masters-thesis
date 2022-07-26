import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.integrate import trapz
from scipy.integrate import quad
from scipy.optimize import root
from scipy.integrate import solve_bvp
from scipy.special import spherical_jn

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

gammaFuchs = np.array([145.31001589825118, 146.10492845786965, 146.97933227344993, 147.8060413354531, 148.93481717011127, 149.7933227344992, 150.82670906200318, 151.65341812400635, 152.4960254372019, 153.3227344992051])
sigmaFuchs = np.array([0.05611510791366906, 0.11330935251798561, 0.15971223021582734, 0.20503597122302158, 0.2870503597122302, 0.39172661870503594, 0.46402877697841727, 0.5924460431654676, 0.6787769784172661, 0.8028776978417266])
sigmaFuchsPoint = np.array([0.04964028776978417, 0.10251798561151079, 0.1510791366906475, 0.1910071942446043, 0.2719424460431655, 0.3755395683453237, 0.44676258992805756, 0.570863309352518, 0.6550359712230216, 0.7780575539568345])
errorFuchsmin = np.subtract(sigmaFuchs,sigmaFuchsPoint)
errorFuchsmax = errorFuchsmin
errorFuchs = [errorFuchsmin, errorFuchsmax]
plt.figure(figsize=(9,5.5))
plt.errorbar(gammaFuchs,sigmaFuchs,yerr=errorFuchs,fmt="o");
plt.xlabel(r"$E_\gamma$ [GeV]")
plt.ylabel(r"$\sigma$ [mb]")

gammaMazzucato = [146.5, 146.6, 147.5, 147.6, 148.5, 150.5, 152.4, 154.1, 154.6, 159.0, 167.1, 169.2]
sigmaMazzucato = [0.25, 0.30, 0.30, 0.20, 0.29, 0.52, 0.86, 1.34, 1.16, 2.44, 5.42, 6.01]
errorMazzucatomin = [0.12, 0.14, 0.09, 0.08, 0.07, 0.08, 0.12, 0.20, 0.16, 0.20, 0.34, 0.58]
errorMazzucatomax = errorMazzucatomin
errorMazzucato = [errorMazzucatomin, errorMazzucatomax]

plt.scatter(gammaMazzucato,sigmaMazzucato);
plt.errorbar(gammaMazzucato,sigmaMazzucato,yerr=errorMazzucato,fmt="o");
plt.xlabel(r"$E_\gamma$ [GeV]")
plt.ylabel(r"$\sigma$ [mb]")

gammaBergstrom = [141.55172413793105, 142.41379310344828, 143.27586206896552, 144.13793103448276, 145, 145.86206896551724, 146.72413793103448, 147.5287356321839, 148.39080459770116, 149.2528735632184, 150.05747126436782, 150.91954022988506, 151.66666666666666, 152.4712643678161, 153.27586206896552, 154.08045977011494, 154.88505747126436, 155.632183908046, 156.4367816091954, 157.29885057471265, 158.04597701149424, 158.85057471264366, 159.5977011494253, 160.3448275862069, 161.09195402298852, 161.89655172413794, 162.64367816091954, 163.44827586206895, 164.19540229885058]
sigmaBergstrom = [0.025684931506849314, 0.059931506849315065, 0.09417808219178081, 0.1626712328767123, 0.1797945205479452, 0.2568493150684931, 0.2868150684931507, 0.3553082191780822, 0.41952054794520544, 0.4708904109589041, 0.547945205479452, 0.6035958904109588, 0.6592465753424657, 0.7320205479452054, 0.8390410958904109, 0.9375, 1.0273972602739725, 1.1130136986301369, 1.2200342465753424, 1.3398972602739725, 1.4683219178082192, 1.5582191780821917, 1.6780821917808217, 1.7851027397260273, 1.9135273972602738, 2.033390410958904, 2.127568493150685, 2.281678082191781, 2.4529109589041096]
sigmaBergstromPoint = [0, 0.05565068493150685, 0.07705479452054795, 0.14126712328767121, 0.1583904109589041, 0.23972602739726026, 0.2696917808219178, 0.3467465753424657, 0.4066780821917808, 0.4537671232876712, 0.5308219178082192, 0.5821917808219178, 0.646404109589041, 0.7148972602739726, 0.821917808219178, 0.9160958904109588, 1.0059931506849316, 1.0873287671232876, 1.2029109589041096, 1.3142123287671232, 1.4426369863013697, 1.5368150684931505, 1.6481164383561644, 1.7508561643835616, 1.8835616438356164, 1.9991438356164382, 2.0976027397260273, 2.2517123287671232, 2.422945205479452]

sigmaBergstromErrorMin = np.subtract(sigmaBergstrom,sigmaBergstromPoint)
sigmaBergstromErrorMax = sigmaBergstromErrorMin
sigmaErrorBergstrom = [sigmaBergstromErrorMin, sigmaBergstromErrorMax]

plt.scatter(gammaBergstrom,sigmaBergstrom);
plt.errorbar(gammaBergstrom,sigmaBergstrom,yerr=sigmaErrorBergstrom,fmt="o");
plt.xlabel(r"$E_\gamma$ [GeV]")
plt.ylabel(r"$\sigma$ [mb]")

diffcrossAngleBecks = [11.356073211314477, 19.217970049916808, 27.371048252911816, 34.65058236272879, 42.803660565723796, 50.374376039933445, 58.23627287853578, 66.0981697171381, 73.96006655574044, 82.11314475873544, 89.6838602329451, 97.54575707154743, 105.40765391014976, 113.26955074875208, 121.13144758735442, 128.99334442595674, 137.14642262895177, 144.7171381031614, 152.57903494176372, 160.44093178036607, 168.59400998336108]
diffcrossBecks = [0.04642857142857143, 0.07976190476190477, 0.07678571428571429, 0.1005952380952381, 0.09464285714285715, 0.14642857142857144, 0.10833333333333334, 0.14047619047619048, 0.1267857142857143, 0.12202380952380953, 0.14583333333333334, 0.13273809523809524, 0.14821428571428572, 0.13630952380952382, 0.12916666666666668, 0.09166666666666667, 0.18214285714285716, 0.10476190476190478, 0.23750000000000002, 0.15833333333333335, 0.07440476190476192]
diffcrossErrorminBecks = [0.028571428571428574, 0.06071428571428572, 0.061309523809523814, 0.08452380952380953, 0.08154761904761905, 0.13095238095238096, 0.09642857142857143, 0.1285714285714286, 0.1142857142857143, 0.11011904761904763, 0.13392857142857145, 0.11964285714285715, 0.13333333333333336, 0.12142857142857144, 0.11250000000000002, 0.07678571428571429, 0.15714285714285717, 0.08154761904761905, 0.19464285714285717, 0.11190476190476191, 0.028571428571428574]
diffcrossErrormaxBecks = diffcrossErrorminBecks
errorBecks = [diffcrossErrorminBecks, diffcrossErrormaxBecks]
