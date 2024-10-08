from math import pi
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("dark_background")

dt = 0.001 # sampling time
t = np.arange(0, 0.5, dt)
N = len(t) # number of data

# The signal
f = np.sin(2*pi*50*t) + np.sin(2*pi*120*t) # Sum of 2 frequencies

# Plot it
plt.clf()
plt.plot(t, f)
plt.show()


freq = (1/(dt*N)) * np.arange(N) # Create x-axis of frequencies in Hz
idx_freq = np.arange(1, np.floor(N/2), dtype='int') # Only plot the first half of freqs

fhat = np.fft.fft(f)
psd = np.real( fhat * np.conj(fhat) ) / N
plt.plot(freq[idx_freq], psd[idx_freq])
plt.show()



