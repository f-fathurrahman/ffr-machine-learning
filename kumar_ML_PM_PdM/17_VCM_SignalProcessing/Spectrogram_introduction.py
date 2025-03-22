# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chapter: Vibration-based Condition Monitoring: Signal Processing and Feature Extraction
#
#
# # Topic:  Time-frequency decomposition

# %%
# generate signal
import numpy as np, matplotlib.pyplot as plt

fs = 1000 # 1000 Hz 
dt = 1.0/fs
t = np.arange(0,1,dt) 

y1 = np.sin(2*np.pi*20*t ) # 20Hz component
y2 = np.sin(2*np.pi*80*t) # 80Hz component
y = np.hstack((y1,y2)) # sampled signal
t = np.hstack((t, t[-1]+t))

plt.figure(figsize = (20, 3))
plt.plot(t*1000, y, '-*',color='black', label='Samples')
plt.ylabel('Amplitude (g)', fontsize=25), plt.xlabel('Time (ms)', fontsize=25)
plt.grid()

# %%
# spectrogram
from scipy import signal

f, t, Sxx = signal.stft(y, fs)

plt.figure(figsize=(8,4))
plt.pcolormesh(t, f, np.abs(Sxx), shading='gouraud')
plt.colorbar()
plt.ylabel('Frequency [Hz]', fontsize=25), plt.xlabel('Time [sec]', fontsize=25)
plt.show()
