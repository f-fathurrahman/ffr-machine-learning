import scipy.io
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.style.use("ggplot")

mat_data = scipy.io.loadmat("../../../DATA/coin_data.mat")

toss_data = mat_data["toss_data"].flatten()
toss_data2 = mat_data["toss_data2"].flatten()
big_data = mat_data["big_data"].flatten()

# Plot the prior density
α = 50;
β = 50;

print("Prior parameters: α: %g, β: %g" % (α, β))

r = np.linspace(0.0, 1.0, 501)

mydist = scipy.stats.beta(α, β)

plt.plot(r, mydist.pdf(r))
plt.xlabel("r")
plt.ylabel("p(r)")
plt.savefig("IMG_coin_01_dist.pdf")

# Incorporate the data one toss at a time
post_α = α
post_β = β
toss2char = ["T", "H"]
toss_string = ""

betapdf = scipy.stats.beta.pdf # shortcut

for i in range(len(toss_data)):
    #
    plt.clf()
    #mydist = scipy.stats.beta(post_α, post_β)
    plt.plot(r, betapdf(r, post_α, post_β), linestyle="--", label="Prev posterior")
    #
    post_α = post_α + toss_data[i]
    post_β = post_β + 1.0 - toss_data[i]
    print("Data: %3d α = %f, β = %f" % (i+1, post_α, post_β))
    #mydist = scipy.stats.beta(post_α, post_β)
    plt.plot(r, betapdf(r, post_α, post_β), label="New posterior")
    #
    plt.xlabel("r")
    plt.ylabel("p(r|...)")
    plt.legend(loc="upper left")
    plt.grid(True)
    #
    toss_string = toss_string + toss2char[toss_data[i]]
    title = "Posterior after {:d} tosses: {:s}".format(i+1, toss_string)
    plt.title(title)
    #
    filesave = "IMG_coin_02_toss_data_{:04d}.png".format(i+1)
    plt.savefig(filesave, dpi=150)


# Incorporate another 10 data
plt.clf()
plt.plot(r, betapdf(r, post_α, post_β), linestyle='--', label="Posterior after 10")
N = len(toss_data2)
post_α = post_α + sum(toss_data2)
post_β = post_β + N - sum(toss_data2)
plt.plot(r, betapdf(r, post_α, post_β), label="Posterior after 20")
plt.xlabel("r")
plt.ylabel("p(r|...)")
plt.legend(loc="upper left")
plt.grid(True)
title = "Posterior after adding 10 more tosses"
plt.title(title)
filesave = "IMG_coin_02_toss_data_0020.png"
plt.savefig(filesave, dpi=150)



#
# Incorporate another 1000 data
#
plt.clf()
plt.plot(r, betapdf(r, post_α, post_β), linestyle='--', label="Posterior after 20")
N = len(big_data)
post_α = post_α + sum(big_data)
post_β = post_β + N - sum(big_data)
plt.plot(r, betapdf(r, post_α, post_β), label="Posterior after 1020")
plt.xlabel("r")
plt.ylabel("p(r|...)")
plt.legend(loc="upper left")
plt.grid(True)
title = "Posterior after adding 1000 more tosses"
plt.title(title)
filesave = "IMG_coin_02_toss_data_1020.png"
plt.savefig(filesave, dpi=150)