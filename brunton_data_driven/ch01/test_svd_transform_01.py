import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["figure.figsize"] = [16, 8]
plt.rcParams.update({"font.size": 18})

# Prepare points for a sphere
u = np.linspace(-np.pi, np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

# Plot sphere
fig = plt.figure()
ax1 = fig.add_subplot(121, projection="3d")
# Plot the surface
surf1 = ax1.plot_surface(
    x, y, z, cmap="jet", alpha=0.6,
    facecolors=plt.cm.jet(z),
    linewidth=0.5, rcount=30, ccount=30)
box3d_aspect_ratio = (np.ptp(x), np.ptp(y), np.ptp(z))
ax1.set_box_aspect(box3d_aspect_ratio)
surf1.set_edgecolor("k")
ax1.set_xlim3d(-2, 2)
ax1.set_ylim3d(-2, 2)
ax1.set_zlim3d(-2, 2)

# Prepare transformation matrix

def prepare_transform_matrix(thetax, thetay, thetaz):
    # Define
    #theta = np.array([np.pi/15, -np.pi/9, -np.pi/20])
    theta = np.array([thetax, thetay, thetaz])
    
    # A diagonal quantity
    Sigma = np.diag([1.0, 1.5, 1.0]) # scale x, then y, then z

    # Rotation about x axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta[0]), -np.sin(theta[0])],
                   [0, np.sin(theta[0]), np.cos(theta[0])]])

    # Rotation about y axis
    Ry = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                   [0, 1, 0],
                   [-np.sin(theta[1]), 0, np.cos(theta[1])]])

    # Rotation about z axis
    Rz = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                   [np.sin(theta[2]), np.cos(theta[2]), 0],
                   [0, 0, 1]])

    # Rotate and scale
    X = Rz @ Ry @ Rx @ Sigma

    return X



X = prepare_transform_matrix(0, 0, 0)
xR = np.zeros_like(x)
yR = np.zeros_like(y)
zR = np.zeros_like(z)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        vec = [x[i,j], y[i,j], z[i,j]]
        vecR = X @ vec # transform this point
        xR[i,j] = vecR[0]
        yR[i,j] = vecR[1]
        zR[i,j] = vecR[2]
        
ax2 = fig.add_subplot(122, projection="3d")
surf2 = ax2.plot_surface(
    xR, yR, zR, cmap="jet", alpha=0.6,
    linewidth=0.5, facecolors=plt.cm.jet(z),
    rcount=30, ccount=30)
ax2.set_box_aspect(box3d_aspect_ratio)
surf2.set_edgecolor("k")
ax2.set_xlim3d(-2, 2)
ax2.set_ylim3d(-2, 2)
ax2.set_zlim3d(-2, 2)
plt.show()

