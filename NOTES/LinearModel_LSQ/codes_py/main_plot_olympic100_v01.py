# Plot olympic100m data using Matplotlib

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-darkgrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14}
)

DATA_PATH = "../../../DATA/olympic100m.txt"
data = np.loadtxt(DATA_PATH, delimiter=",")

t = data[:,1]
x = data[:,0]

plt.clf()
plt.plot(x, t, marker="o", label="data")
plt.grid(True)
plt.legend()
plt.title("olympic100m")
plt.xlabel("Year")
plt.ylabel("Winning time (seconds)")
plt.savefig("IMG_data_olympic100m.png", dpi=150)
plt.savefig("IMG_data_olympic100m.pdf")
