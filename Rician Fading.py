# This python script is used to generate rayleigh channel coefficients
import numpy as np
import matplotlib.pyplot as plt

# Sinusoidal waveform generation
t = np.linspace(1, 100, 750)
x_volts = 20*np.sin(t/(2*np.pi))
x_watts = x_volts ** 2
x_db = 10 * np.log10(x_watts)

# Parameters for simulation
v = 60 # velocity (meters per second)
center_freq = 100e6 # RF 100 MHz
Fs = 2e5 # sample rate 0.2 MHz
N = 1000 # Total numbers of sine waves
pi = 3.14
fd = v*center_freq/3e8 # Doppler frequency shift (maximum)
print("Doppler frequency shift (Max.):", fd)
t = np.arange(0, 1, 1/Fs) # time vector. (start, stop, step)
x = np.zeros(len(t))
y = np.zeros(len(t))
for i in range(N):
  alpha = (np.random.rand() - 0.5) * 2 * pi
  phi = (np.random.rand() - 0.5) * 2 * pi
  x = x + np.random.randn() * np.cos(2 * pi * fd * t * np.cos(alpha) + phi)
  y = y + np.random.randn() * np.sin(2 * pi * fd * t * np.cos(alpha) + phi)

z = (1/np.sqrt(N)) * (x + 1j*y) # This is channel response used to convolve with transmitted data or signal
z_mag = np.abs(z) # Used in plot
z_mag_dB = 10*np.log10(z_mag) # convert to dB

# Convolve sinusoidal waveform with Rayleigh Fading channel
y3 = np.convolve(z, x_volts)

# Plots
figure, axis = plt.subplots(2, 2)
axis[0, 0].plot(x_volts)
axis[0, 0].set_title("Pure sine wave signal")
axis[0, 1].plot(z)
axis[0, 1].set_title("Rayleigh Channel response")
axis[1, 0].plot(z_mag_dB)
axis[1, 0].set_title("Rayleigh Channel response (dB)")
axis[1, 1].plot(y3)
axis[1, 1].set_title("Convolved sine wave signal")
plt.tight_layout()
plt.show()