import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

data = np.loadtxt("../../../DATA/olympic100m.txt", delimiter=",")

t = data[:,1]
x = data[:,0]

plt.clf()
plt.plot(x, t, marker="o", label="data")
plt.grid(True)
plt.legend()
plt.xlabel("year")
plt.ylabel("seconds")
plt.savefig("IMG_data_olympic100m.png", dpi=150)
plt.savefig("IMG_data_olympic100m.pdf")
