Nfirst = 1440
plt.clf()
plt.plot(range(Nfirst), temperature[:Nfirst])
plt.savefig("IMG_first_10days.png", dpi=150)

