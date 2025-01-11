"""
utils.py: Helper functions for solver.py.
"""

import numpy as np
import scipy.fft as fft
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
jax.config.update('jax_enable_x64', True)


def check_bc(psi0, t0, f, rtol=1e-5, atol=1e-8):
    '''Checks whether boundary conditions are initially satisfied or not.'''

    if f is not None:
        f1, f2 = f
        if not np.isclose(psi0[0], f1(t0), rtol, atol):
            raise ValueError(f'Left bdry. cond. not satisfied with rtol={rtol:.2e}, atol={atol:.2e}.')
        if not np.isclose(psi0[-1], f2(t0), rtol, atol):
            raise ValueError(f'Right bdry. cond. not satisfied with rtol={rtol:.2e}, atol={atol:.2e}.')


def check_bdry(t, x1, x2):
    '''Checks whether boundaries are valid or not.'''

    L = x2(t) - x1(t)
    cond = L < 0
    if np.any(cond):
        i = np.argmax(cond)
        raise ValueError(f'Boundary definition fails for t={t[i]:.2f}, negative domain length.')


def boundaries(x1, x2, V, BC, f):
    '''Computes helper functions based on the boundary conditions.'''

    if BC == 'periodic':
        if f is not None:
            raise ValueError('Periodic BCs do not support inhomogeneous terms.')
        def up(t, x):
            return 0j
        expT = periodic
    elif BC == 'dirichlet':
        if f is not None:
            def f1(t):
                return f[0](t) + 0j
            def f2(t):
                return f[1](t) + 0j
            f1_t = grad(f1, holomorphic=True)
            f2_t = grad(f2, holomorphic=True)
            @jit
            def up(t, x):
                M1 = jnp.array([
                    [-1,             1],
                    [3*x2(t), -3*x1(t)]
                ]) / (6 * (x2(t) - x1(t)))
                v1 = jnp.array([
                    V(t, x1(t)) * f1(t) - 1j * f1_t(t),
                    V(t, x2(t)) * f2(t) - 1j * f2_t(t)
                ])
                A, B = M1 @ v1
                M2 = jnp.array([
                    [-1,         1],
                    [x2(t), -x1(t)]
                ]) / (x2(t) - x1(t))
                v2 = jnp.array([
                    f1(t) - A*x1(t)**3 - B*x1(t)**2,
                    f2(t) - A*x2(t)**3 - B*x2(t)**2
                ])
                C, D = M2 @ v2
                return A*x**3 + B*x**2 + C*x + D
        else:
            def up(t, x):
                return 0j
        expT = dirichlet
    else:
        raise ValueError('Boundary conditions must be "periodic" or "dirichlet".')
    @jit
    def up_t(t, x):
        func = grad(up, holomorphic=True, argnums=0)
        return vmap(func, (None, 0))(t, x)
    @jit
    def up_xx(t, x):
        func = grad(grad(up, holomorphic=True, argnums=1), holomorphic=True, argnums=1)
        return vmap(func, (None, 0))(t, x)
    return expT, up, up_t, up_xx


def periodic(psi, dt, x):
    '''Kinetic energy time step for periodic BC.'''

    Nx = len(x)
    dx = np.diff(x)[0]
    k = 2 * np.pi * fft.fftfreq(Nx, dx)
    psi_f = fft.fft(psi)
    psi_f = psi_f * np.exp(- 1j * dt * k**2 / 2)
    psi = fft.ifft(psi_f)
    return psi


def dirichlet(psi, dt, x):
    '''Kinetic energy time step for dirichlet BC.'''

    Nx = len(x)
    dx = np.diff(x)[0]
    k = np.pi * np.arange(1, Nx + 1) / (Nx * dx)
    psi_f = fft.dst(psi, type=1)
    psi_f = psi_f * np.exp(- 1j * dt * k**2 / 2)
    psi = fft.idst(psi_f, type=1)
    return psi


def cutoff_periodic(wf_f, k, fraction=.999):
    '''Noise threshold for periodic BC.'''

    n = len(k)

    k_neg   = k[-1:n//2+n%2-1:-1]
    f_neg   = wf_f[-1:n//2+n%2-1:-1]
    pow_neg = np.abs(f_neg)**2
    cum_neg = np.cumsum(pow_neg) / np.sum(pow_neg)
    i = np.argmax(cum_neg >= fraction)
    kmin = k_neg[i]

    k_pos   = k[:n//2+n%2]
    f_pos   = wf_f[:n//2+n%2]
    pow_pos = np.abs(f_pos)**2
    cum_pos = np.cumsum(pow_pos) / np.sum(pow_pos)
    i = np.argmax(cum_pos >= fraction)
    kmax = k_pos[i]

    return kmin, kmax


def cutoff_dirichlet(wf_f, k, fraction=.999):
    '''Noise threshold for dirichlet BC.'''

    power = np.abs(wf_f)**2
    cumulative_power = np.cumsum(power) / np.sum(power)
    i = np.argmax(cumulative_power >= fraction)
    return k[i]


def energy(psi, uh, H_up, x, V, norm_2, BC):
    '''Average energy and standard deviation.'''

    if BC == 'periodic':
        def H(wf):
            Nx = len(x)
            dx = np.diff(x)[0]
            k = 2 * np.pi * fft.fftfreq(Nx, dx)
            wf_f = fft.fft(wf)
            kmin, kmax = cutoff_periodic(wf_f, k)
            k_filter = (k >= kmin) * (k <= kmax)
            T_wf_f = k**2/2 * wf_f * k_filter
            T_wf = fft.ifft(T_wf_f)
            return T_wf + V * wf
    elif BC == 'dirichlet':
        def H(wf):
            Nx = len(x)
            dx = np.diff(x)[0]
            k = np.pi * np.arange(1, Nx + 1) / (Nx * dx)
            wf_f = fft.dst(wf, type=1)
            kc = cutoff_dirichlet(wf_f, k)
            k_filter = k <= kc
            T_wf_f = k**2/2 * wf_f * k_filter
            T_wf = fft.idst(T_wf_f, type=1)
            return T_wf + V * wf

    H_psi = H(uh) + H_up
    E = np.trapezoid(psi.conj()*H_psi, x).real / norm_2
    H2_psi = H(H_psi)
    E2 = np.trapezoid(psi.conj()*H2_psi, x).real / norm_2
    dE = np.sqrt(max(0, E2 - E**2))
    return E, dE
