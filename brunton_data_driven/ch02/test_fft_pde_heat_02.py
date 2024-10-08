# Persamaan difusi (kalor):
# $$
# u_{t} = \alpha^2 u_{xx}
# $$

# Dengan menggunakan transformasi Fourier $\mathcal{F}\left[ u(t,x) \right] = \hat{u}(t,\omega)$, persamaan ini dapat diubah menjadi:
# $$
# \hat{u}_{t} = -\alpha^2 \omega^2 \hat{u}
# $$

# PDE telah diubah menjadi ODE (dalam koordinat frekuensi spasial).
# Kita perlu menyelesaikan ODE ini untuk sebanyak `Nfreq=Npoints`, sejumlah titik spasial yang digunakan untuk diskritisasi $u$.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

import matplotlib
matplotlib.style.use("dark_background")

from math import pi

from mpl_toolkits.mplot3d import axes3d
#plt.rcParams['figure.figsize'] = [12, 12]
#plt.rcParams.update({'font.size': 18})

α = 1    # Thermal diffusivity constant
L = 100  # Length of domain
N = 1000 # Number of discretization points
dx = L/N
x = np.arange(-L/2,L/2,dx) # Define x domain

# ?np.fft.fftfreq

N1 = 10
x1 = np.linspace(0.0, 2.0, N1);
x1

np.fft.fftfreq(N1,  d=x1[1]-x1[0])

np.fft.fftfreq(N1)

# Define discrete wavenumbers
κ = 2 * pi * np.fft.fftfreq(N, d=dx)

# Initial condition
u0 = np.zeros_like(x)
u0[int((L/2 - L/10)/dx):int((L/2 + L/10)/dx)] = 1
u0hat = np.fft.fft(u0)

plt.plot(x, u0);

# SciPy's odeint function doesn't play well with complex numbers, so we recast 
# the state u0hat from an N-element complex vector to a 2N-element real vector
u0hat_ri = np.concatenate( (u0hat.real, u0hat.imag) )

# Simulate in Fourier frequency domain
dt = 0.1
t = np.arange(0,10,dt)

def rhsHeat(uhat_ri, t, κ, α):
    uhat = uhat_ri[:N] + (1j) * uhat_ri[N:]
    d_uhat = -α**2 * ( np.power(κ,2) ) * uhat
    d_uhat_ri = np.concatenate((d_uhat.real,d_uhat.imag)).astype('float64')
    return d_uhat_ri

# FIXME: algoritma apa yang digunakan oleh `odeint`?

uhat_ri = odeint( rhsHeat, u0hat_ri, t, args=(κ,α) )

# Bentuk lagi uhat dalam bilangan kompleks:

uhat = uhat_ri[:,:N] + (1j) * uhat_ri[:,N:]

u = np.zeros_like(uhat)

# Transformasi balik:

for k in range(len(t)):
    u[k,:] = np.fft.ifft(uhat[k,:])

# Ambil bagian real:

u = u.real    

u.shape

plt.plot(x, u[0,:], label="initial")
plt.plot(x, u[-1,:], label="last");
