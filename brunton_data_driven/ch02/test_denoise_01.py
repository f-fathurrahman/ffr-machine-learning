import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams.update({'font.size': 18})

# Create a simple signal with two frequencies
dt = 0.001
t = np.arange(0,1,dt)
f = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t) # Sum of 2 frequencies
f_clean = f
f = f + 2.5*np.random.randn(len(t))              # Add some noise

plt.clf()
plt.plot(t, f, color='r', linewidth=1.5, label='Noisy')
plt.plot(t, f_clean, color='k', linewidth=2, label='Clean')
plt.xlim(t[0], t[-1])
plt.xlabel("Time")
plt.legend()
plt.show()
plt.savefig("IMG_noisy_vs_clean.png", dpi=150)


## Compute the Fast Fourier Transform (FFT)
n = len(t)
fhat = np.fft.fft(f,n)                     # Compute the FFT
PSD = np.real( fhat * np.conj(fhat) / n )             # Power spectrum (power per freq)
freq = (1/(dt*n)) * np.arange(n)           # Create x-axis of frequencies in Hz
idx_freq = np.arange(1, np.floor(n/2), dtype='int') # Only plot the first half of freqs

## Use the PSD to filter out noise
indices = PSD > 100       # Find all freqs with large power, cutoff=100
PSDclean = PSD * indices  # Zero out all others
fhat = indices * fhat     # Zero out small Fourier coeffs. in Y
ffilt = np.real( np.fft.ifft(fhat) ) # Inverse FFT for filtered time signal

plt.clf()
plt.plot(t, f_clean, color='k', linewidth=1.5,label='Clean')
plt.plot(t, ffilt, color='b', linewidth=2,label='Filtered')
plt.xlim(t[0], t[-1])
plt.xlabel("Time")
plt.legend()
plt.savefig("IMG_clean_vs_filtered.png", dpi=150)
plt.show()

plt.clf()
plt.plot(freq[idx_freq], PSD[idx_freq], color='r', linewidth=2, label='Noisy')
plt.plot(freq[idx_freq], PSDclean[idx_freq], color='b', linewidth=1.5, label='Filtered')
plt.xlim(freq[idx_freq[0]], freq[idx_freq[-1]])
plt.xlabel("Frequency")
plt.legend()
plt.savefig("IMG_PSD_clean_vs_filtered_01.png", dpi=150)
plt.show()


