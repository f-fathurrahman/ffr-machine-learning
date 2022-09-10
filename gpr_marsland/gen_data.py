import numpy as np
import matplotlib.pyplot as plt

def my_func(x):
    return np.sin(3*x) + np.cos(2*x)

x = np.linspace(-1.0, 1.0, 100)
y = my_func(x)

plt.clf()
plt.plot(x, y, marker="o")
plt.grid()
plt.savefig("IMG_gen_data.pdf")

np.random.seed(1)
idx_sample = np.random.choice(np.arange(100), size=10)
idx_sample.sort() # to make it easier to plot
print(idx_sample)

x_sample = x[idx_sample]
y_sample = y[idx_sample]

f = open("data2.txt", "w")
for i in range(len(x_sample)):
    print("%18.10f %18.10f" % (x_sample[i], y_sample[i]), file=f)
f.close()

# Sort before
plt.clf()
plt.plot(x_sample, y_sample, marker="o")
plt.grid()
plt.savefig("IMG_gen_data_sample.pdf")