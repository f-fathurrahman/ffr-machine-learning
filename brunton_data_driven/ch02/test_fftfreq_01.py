import numpy as np
from math import pi

n = 12
L = 5.0
dx = L/n
x = np.arange(-L/2, L/2, dx)

# Manual construction of fft frequencies with help of fftshift
κ = (2*pi/L)*np.arange(-n/2, n/2)
print("κ unshifted = ", κ)
κ_shifted = np.fft.fftshift(κ)
print("κ_shifted = ", κ_shifted)

# Using built-in fftfreq
ω = np.fft.fftfreq(n, d=dx) * 2*pi
print("ω = ", ω)

