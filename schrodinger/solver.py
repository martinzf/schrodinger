"""
solver.py: Module for solving the Schrödinger equation.
"""

import numpy as np

from . import wavefunction
from . import utils


def solve(psi0, t, x1, x2, V, BC, f=None, imag=False):
    """
    Solves the Schrödinger equation with time-dependent boundaries.

    Parameters:
        psi0 (np.ndarray[np.complex128]): Initial wavefunction.
        t (np.ndarray[float]): Array of time points for the simulation.
        x1 (Callable[[float], float]): Function specifying the left boundary as a function of time.
        x2 (Callable[[float], float]): Function specifying the right boundary as a function of time.
        V (Callable[[float, np.ndarray[float]], np.ndarray[np.complex128]]): Potential energy function, 
            depends on time and spatial coordinates.
        BC (str): Boundary condition, either 'periodic' or 'dirichlet'.
        f (Optional[Callable[[float], Tuple[float, float]]]): Optional function specifying additional 
            boundary dynamics. Default is None.
        imag (bool): Whether to perform the computation in imaginary time. Default is False.

    Returns:
        Wavefunction: Solution of the wavefunction at each time step.
    """

    # Boundary management
    Nt = len(t)
    Nx = len(psi0)
    psi = np.empty((Nt, Nx), dtype=np.complex128)
    utils.check_bc(psi0, t[0], f)
    utils.check_bdry(t, x1, x2)
    expT, up, up_t, up_xx = utils.boundaries(x1, x2, V, BC, f)

    # Complexification of (t, x)
    t += 0j
    if imag:
        t *= 1j
    dt = np.concatenate(([0], np.diff(t)))
    def x(t):
        return np.linspace(x1(t), x2(t), Nx)+0j

    # Initial conditions
    norm_2 = np.trapezoid(np.abs(psi0)**2, x(t[0]).real)
    psi[0, :] = psi0
    uh = psi0 - up(t[0], x(t[0]))

    # Inhomogeneous (boundary) terms
    H_up = -1/2 * up_xx(t[0], x(t[0])) + V(t[0], x(t[0])) * up(t[0], x(t[0]))

    # Energies
    E, dE = [], []
    E_i, dE_i = utils.energy(psi0, uh, H_up, x(t[0]), V(t[0], x(t[0])), norm_2, BC)
    E.append(E_i)
    dE.append(dE_i)

    # Integration loop
    for i in range(1, Nt):
        # Inhomogeneous (boundary) terms, Euler step
        H_up = -1/2 * up_xx(t[i], x(t[i])) + V(t[i], x(t[i])) * up(t[i], x(t[i]))
        uh += -1j * dt[i] * (H_up - 1j*up_t(t[i], x(t[i])))

        # Homogeneous evolution, exponential integration
        uh = uh * np.exp(- 1j * dt[i] * V(t[i], x(t[i])) / 2)
        uh = expT(uh, dt[i], x(t[i]))
        uh = uh * np.exp(- 1j * dt[i] * V(t[i], x(t[i])) / 2)

        # Normalisation
        psi_i = uh + up(t[i], x(t[i]))
        norm_2 = np.trapezoid(np.abs(psi_i)**2, x(t[i]).real)
        psi[i, :] = psi_i

        # Energies
        E_i, dE_i = utils.energy(psi_i, uh, H_up, x(t[i]), V(t[i], x(t[i])), norm_2, BC)
        E.append(E_i)
        dE.append(dE_i)

    if imag:
        t *= -1j
    t = t.real
    sol = wavefunction.Wavefunction(psi, t, x1, x2, V, E, dE)
    return sol
