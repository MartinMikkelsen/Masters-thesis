# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="dark")
# %% 

m = 469.4  # MeV
E = 1.11  # MeV
U0 = 24.82  # MeV
R = 2.127  # fm
hbar = 6.6e-36
k = 1.001
c = 1.001
A = 1
D = 7.1
r = np.linspace(0, 10, 100)  # fm
r1 = np.linspace(0, R, 100)
r2 = np.linspace(R, 10, 100)

u1 = A*np.sin(k*r1)
u2 = D*np.exp(-c*r2)
u = u1+u2


plt.text(R, -2.5, '$-U_0$', color='b')
plt.vlines(R, 0, -2, color='b')
b = sns.lineplot(x=r1, y=-2, color='b')
a = sns.lineplot(x=r1, y=u1, color='maroon', linewidth=1.6)
plt.xlim([-0.1, 8])
plt.ylim([-3, 3])
plt.yticks([0])
plt.ylabel("U(r) [Arb. units]")
plt.xlabel("r [fm]")
sns.lineplot(x=r2, y=u2, color='r', linewidth=1.6)
plt.text(0.5, 1.2, '$\propto \sin(kr)$', color='maroon')
plt.text(2.5, 1, '$\propto \exp(-\kappa r)$', color='r')
plt.hlines(0, 0, 8, color='k')
plt.legend(['Potential'])
plt.savefig('DeuteronWavefunction.pdf')
fig = plt.figure()
