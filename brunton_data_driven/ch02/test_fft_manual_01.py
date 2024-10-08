from math import pi
import numpy as np

np.random.seed(1234)

N = 6
f = np.random.rand(N)

# Forwared FFT using built-in fft function
F_fft = np.fft.fft(f)

# frequencies
ω = np.zeros(N)
for k in range(N):
    ω[k] = 2*pi/N * k
# compare with fftfreq

# indices j and k start from 0 to (N-1)
# No need for modification for Python as the indexing starts from 0

# Forward FFT
F = np.zeros(N, dtype=np.complex128)
for k in range(N):
    for j in range(N):
        F[k] += f[j] * np.exp(-1j * 2*pi/N * k * j)


# Inverse FFT
f_recov = np.zeros(N, dtype=np.complex128)
for k in range(N):
    for j in range(N):
        f_recov[k] += F[j] * np.exp(1j * 2*pi/N * k * j)
    f_recov[k] *= (1/N)


