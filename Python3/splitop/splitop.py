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

'''


class Param:
    """Container for holding all simulation parameters."""

    def __init__(self,
                 xmax: float,
                 res: int,
                 dt: float,
                 timesteps: int,
                 im_time: bool) -> None:

        self.xmax = xmax
        self.res = res
        self.dt = dt
        self.timesteps = timesteps
        self.im_time = im_time

        self.dx = 2 * xmax / res
        self.x = np.arange(-xmax + xmax / res, xmax, self.dx)
        self.dk = pi / xmax
        self.k = np.concatenate((np.arange(0, res / 2),
                                 np.arange(-res / 2, 0))) * self.dk


class Operators:
    """Container for holding operators and wavefunction coefficients."""

    def __init__(self, res: int) -> None:

        self.V = np.empty(res, dtype=complex)
        self.R = np.empty(res, dtype=complex)
        self.K = np.empty(res, dtype=complex)
        self.wfc = np.empty(res, dtype=complex)


def init(par: Param, voffset: float, wfcoffset: float) -> Operators:
    """Initialize the wavefunction coefficients and the potential."""
    opr = Operators(len(par.x))
    opr.V = 0.5 * (par.x - voffset) ** 2
    opr.wfc = np.exp(-((par.x - wfcoffset) ** 2) / 2, dtype=complex)
    if par.im_time:
        opr.K = np.exp(-0.5 * (par.k ** 2) * par.dt)
        opr.R = np.exp(-0.5 * opr.V * par.dt)
    else:
        opr.K = np.exp(-0.5 * (par.k ** 2) * par.dt * 1j)
        opr.R = np.exp(-0.5 * opr.V * par.dt * 1j)
    return opr


def split_op(par: Param, opr: Operators) -> None:

    for i in range(par.timesteps):

        # Half-step in real space
        opr.wfc *= opr.R

        # FFT to momentum space
        opr.wfc = np.fft.fft(opr.wfc)

        # Full step in momentum space
        opr.wfc *= opr.K

        # iFFT back
        opr.wfc = np.fft.ifft(opr.wfc)

        # Final half-step in real space
        opr.wfc *= opr.R

        # Density for plotting and potential
        density = np.abs(opr.wfc) ** 2

        # Renormalizing for imaginary time
        if par.im_time:
            renorm_factor = sum(density) * par.dx
            opr.wfc /= sqrt(renorm_factor)

        # Outputting data to file. Plotting can also be done in a
        # similar way. This is set to output exactly 100 files, no
        # matter how many timesteps were specified.
        if i % (par.timesteps // 100) == 0:
            filename = "output/output{}.dat".format(str(i).rjust(5, str(0)))
            with open(filename, "w") as outfile:
                # Outputting for gnuplot. Any plotter will do.
                for j in range(len(density)):
                    template = "{}\t{}\t{}\n".format
                    line = template(par.x[j], density[j].real, opr.V[j].real)
                    outfile.write(line)
            print("Outputting step: ", i + 1)


def calculate_energy(par: Param, opr: Operators) -> float:
    """Calculate the energy <Psi|H|Psi>."""
    # Creating real, momentum, and conjugate wavefunctions.
    wfc_r = opr.wfc
    wfc_k = np.fft.fft(wfc_r)
    wfc_c = np.conj(wfc_r)

    # Finding the momentum and real-space energy terms
    energy_k = 0.5 * wfc_c * np.fft.ifft((par.k ** 2) * wfc_k)
    energy_r = wfc_c * opr.V * wfc_r

    # Integrating over all space
    energy_final = sum(energy_k + energy_r).real

    return energy_final * par.dx


def main() -> None:
    par = Param(20.0, 256, 0.05, 100, False)

    # Starting wavefunction slightly offset so we can see it change
    opr = init(par, 0.0, -1.00)
    split_op(par, opr)

    energy = calculate_energy(par, opr)
    print("Energy is: ", energy)


if __name__ == "__main__":
    main()
'''
