import numpy as np
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

x, y = np.mgrid[-2.0:2.0:.01, -2.0:2.0:.01]
pos = np.dstack((x, y))

#rv = multivariate_normal([0.5, 0.0], [[2.0, 0.3], [0.3, 0.5]])
rv = multivariate_normal([0.5, 1.0], [[0.1, 0.0], [0.0, 0.1]])

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.contourf( x, y, rv.pdf(pos) )
plt.gca().set_aspect("equal", "box")
plt.savefig("IMG_mvn_countourf_v01.pdf")
