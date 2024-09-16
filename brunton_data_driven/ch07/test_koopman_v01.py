import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate

rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]

mu = -0.05
lamb = -1
A = np.array([[mu,0,0],[0,lamb,-lamb],[0,0,2*mu]]) # Koopman linear dynamics
D,T = np.linalg.eig(A)
slope_stab_man = T[2,2]/T[1,2] # slope of stable subspace (green)


