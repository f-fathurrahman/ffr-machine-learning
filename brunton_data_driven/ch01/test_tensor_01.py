# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc
from IPython.display import HTML
# %matplotlib inline

plt.rcParams['figure.figsize'] = [8,8]
# plt.rcParams.update({'font.size': 18})
# plt.rcParams['animation.html'] = 'jshtml'

x = np.arange(-5, 5.01, 0.1)
y = np.arange(-6, 6.01, 0.1)
t = np.arange(0, 10*np.pi+0.1, 0.1)

# An example quantity that varies on space and time
X,Y,T = np.meshgrid(x,y,t)
A = np.exp(-(X**2 + 0.5*Y**2)) * np.cos(2*T) + \
    (np.divide(np.ones_like(X),np.cosh(X)) * np.tanh(X) * np.exp(-0.2*Y**2)) * np.sin(T)

"""
fig = plt.figure()
ax = fig.add_subplot()
# at t=0
for itime in range(len(t)):
    ax.cla()
    ax.pcolormesh(X[:,:,itime], Y[:,:,itime], A[:,:,itime], vmin=-1, vmax=1, shading='auto')
    filename = "IMG_funcxyt_{:04d}.png".format(itime)
    plt.savefig(filename, dpi=150)
    print(f"itime = {itime} done")
"""

from tensorly.decomposition import parafac
w, AA = parafac(A, 2)
plt.clf()
fig, axs = plt.subplots(3,1)
axs[0].plot(y, AA[0], linewidth=2)
axs[1].plot(x, AA[1], linewidth=2)
axs[2].plot(t, AA[2], linewidth=2)
plt.savefig("IMG_parafac_A.png", dpi=150)


"""
def init():
    pcm.set_array(np.array([]))
    return pcm

def animate(iter):
    pcm.set_array(A[:-1,:-1,iter].ravel())
#     print('Frame ' + str(iter))
    return pcm

anim = animation.FuncAnimation(
    fig, animate,
    init_func=init,
    frames=len(t),
    interval=50,
    blit=False, repeat=False)
HTML(anim.to_jshtml())
"""