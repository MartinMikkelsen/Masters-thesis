import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.linalg import inv

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
    plt.savefig(image_path(fig_id) + ".png", format='png')

#Relative coorinates

def JacobiU3(m1,m2,m3):
    U_J = np.array([[1,-1,0],[m1/(m1+m2),m2/(m1+m2),-1], [m1/(m1+m2+m3), m2/(m1+m2+m3), m3/(m1+m2+m3)]])
    U_Jinv = inv(U_J)
    return U_J

#Example

A = JacobiU3(936,925,923)
B = inv(JacobiU3(936,925,923))

def visualize(r,V):
    plt.quiver([0, 0, 0], [0, 0, 0], r, V , angles='xy', scale_units='xy', scale=1)
    plt.xlim(-1, 1)
    plt.ylim(-1 , 1)
    return plt.show()


r1 = np.array([1,2,3])
r2 = np.array([3,2,1])
r3 = np.array([5,2,3])

r = np.array([r1,r2,r3])
