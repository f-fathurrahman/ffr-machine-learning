import numpy as np
import matplotlib.pyplot as plt

#plt.rcParams['figure.figsize'] = [12, 12]
#plt.rcParams.update({'font.size': 18})


n = 128
L = 30
dx = L/n
x = np.arange(-L/2, L/2, dx, dtype=np.complex128)
f = np.cos(x) * np.exp(-np.power(x,2)/25) # Function
df = -(np.sin(x) * np.exp(-np.power(x,2)/25) + (2/25)*x*f) # Derivative

## Approximate derivative using finite difference
dfFD = np.zeros(len(df), dtype=np.complex128)
for kappa in range(len(df)-1):
    dfFD[kappa] = (f[kappa+1]-f[kappa])/dx

dfFD[-1] = dfFD[-2]

## Derivative using FFT (spectral derivative)
fhat = np.fft.fft(f)
kappa = (2*np.pi/L)*np.arange(-n/2,n/2)
kappa = np.fft.fftshift(kappa) # Re-order fft frequencies
dfhat = kappa * fhat * (1j)
dfFFT = np.real(np.fft.ifft(dfhat))

## Plots
plt.plot(x,df.real,color='k',linewidth=2,label='True Derivative')
plt.plot(x,dfFD.real,'--',color='b',linewidth=1.5,label='Finite Diff.')
plt.plot(x,dfFFT.real,'--',color='r',linewidth=1.5,label='FFT Derivative')
plt.legend()
plt.show()

