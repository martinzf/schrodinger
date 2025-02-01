"""
solver.py: Module for solving the Schrödinger equation.
"""

import numpy as np

from . import wavefunction
from . import utils


def solve(psi0, t, x1, x2, V, BC, imag=False):
    """
    Solves the Schrödinger equation with time-dependent boundaries.

    Parameters:
        psi0 (np.ndarray[np.complex128]): Initial wavefunction.
        t (np.ndarray[float]): Array of time points for the simulation.
        x1 (Callable[[float], float]): Function specifying the left boundary as a function of time.
        x2 (Callable[[float], float]): Function specifying the right boundary as a function of time.
        V (Callable[[float, np.ndarray[float]], np.ndarray[np.complex128]]): Potential energy function, 
            depends on time and spatial coordinates.
        BC (str): Boundary condition, 'periodic', 'dirichlet' or 'pml'.
        imag (bool): Whether to perform the computation in imaginary time. Default is False.

    Returns:
        Wavefunction: Solution of the wavefunction at each time step.
    """

    # Boundary management
    Nt = len(t)
    Nx = len(psi0)
    psi = np.empty((Nt, Nx), dtype=np.complex128)
    utils.check_bc(psi0, BC)
    utils.check_bdry(t, x1, x2)
    N, dx, x, x_t, x_tt = utils.coords(x1, x2, Nx, BC)
    dt = np.concatenate(([0], np.diff(t)))
    if imag:
        dt = -1j * dt

   # Effective potential and momentum step
    Veff = utils.compute_Veff(x, x_t, x_tt, x1, x2, V, BC)
    expT = utils.compute_expT(x, x1, x2, N, BC)

    # Initial conditions
    psi[0, :] = psi0
    if BC == 'pml':
        F = utils.stretch(x(t[0]), x1(t[0]), x2(t[0]))
        scaling = np.exp(-1j * F * x_t(t[0]))
        psi_i = utils.extrapolate(psi0, x(t[0]), x1(t[0]), x2(t[0]), Nx) * scaling
    else:
        scaling = np.exp(-1j * x(t[0]) * x_t(t[0]))
        psi_i = psi0 * scaling

    # Integration loop
    for i in range(1, Nt):
        # Split operator step
        psi_i *= np.exp(- .5j * dt[i] * Veff(t[i]))
        psi_i  = expT(psi_i, t[i], dt[i], dx(t[i]))
        psi_i *= np.exp(- .5j * dt[i] * Veff(t[i]))

        # Save data
        if BC == 'pml':
            F = utils.stretch(x(t[i]), x1(t[i]), x2(t[i]))
            scaling = np.exp(-1j * F * x_t(t[i]))
            psi[i, :] = (psi_i * scaling)[int(utils.D*Nx):-int(utils.D*Nx)]
        else:
            scaling = np.exp(1j * x(t[i]) * x_t(t[i]))
            psi[i, :] = psi_i * scaling

    return wavefunction.Wavefunction(psi, t, x1, x2, V, BC)
