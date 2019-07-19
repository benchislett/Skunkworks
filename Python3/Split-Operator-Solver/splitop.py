from math import pi
from math import sqrt

import numpy as np

# Configure Parameters

XMIN = -20.0
XMAX = 20.0
RES = 256
STEP_SIZE = 0.05 * -1j  # imaginary time
TIMESTEPS = 100


# Init Variables

DX = 2 * XMAX / RES
X = np.arange(XMIN + (XMAX / RES), XMAX, DX)
DK = pi / XMAX
K_I = np.concatenate((np.arange(0, RES / 2), np.arange(-RES / 2, 0))) * DK


# Configure Operators

V = 0.5 * X ** 2
WFC = np.exp(-((X + 1) ** 2) / 2, dtype=complex)


# Init Operators

K = np.exp(-0.5 * (K_I ** 2) * STEP_SIZE * 1j, dtype=complex)
R = np.exp(-0.5 * V * STEP_SIZE * 1j, dtype=complex)


# Split-operator fourier method
for i in range(TIMESTEPS):
    # Half-step in real space
    WFC *= R

    # FFT to momentum space
    WFC = np.fft.fft(WFC)

    # Full step in momentum space
    WFC *= K

    # iFFT back to real space
    WFC = np.fft.ifft(WFC)

    # Half-step in real space
    WFC *= R

    # Density for plotting and potential
    density = np.abs(WFC) ** 2

    # Normalize for imaginary time
    if (np.iscomplex(STEP_SIZE)):
        factor = sum(density) * DX
        WFC /= sqrt(factor)

    # Outputting data to file. Plotting can also be done in a
    # similar way. This is set to output exactly 100 files, no
    # matter how many timesteps were specified.
    if (i % (TIMESTEPS // 100) == 0):
        filename = "output/output{}.dat".format(str(i).zfill(5))
        with open(filename, "w") as outfile:
            # Outputting for gnuplot. Any plotter will do.
            for j in range(len(density)):
                line = "{}\t{}\t{}\n".format(
                    X[j], density[j].real, V[j].real)
                outfile.write(line)
        print("Outputting step: ", i + 1)

# Calculate the energy < Psi | H | Psi >
# Creating real, momentum, and conjugate wavefunctions.
WFC_R = WFC
WFC_K = np.fft.fft(WFC_R)
WFC_C = np.conj(WFC_R)

# Finding the momentum and real-space energy terms
energy_k = 0.5 * WFC_C * np.fft.ifft((K_I ** 2) * WFC_K)
energy_r = WFC_C * V * WFC_R

# Integrating over all space
energy_final = sum(energy_k + energy_r).real

print('Final energy: ', energy_final * DX)
