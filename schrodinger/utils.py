"""
utils.py: Helper numerical functions for solver.py and wavefunction.py.
"""

import numpy as np
import scipy.fft as fft
import finufft
from scipy.interpolate import CubicSpline
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
jax.config.update('jax_enable_x64', True)


# BOUNDARY CONDITION VALIDATION
RTOL = 1e-5
ATOL = 1e-8

# PML PARAMETERS
G  = jnp.pi / 4     # Absorption coefficient exp(1j*G)
S0 = 0              # Absorption strength factor
D  = .1             # Absorbing layer depth / solution domain

# NOISE THRESHOLD (for derivatives)
F = .9995           # Fraction of power spectrum we keep


def check_bc(psi0, BC, rtol=RTOL, atol=ATOL):
    '''Checks whether boundary conditions are initially satisfied or not.'''

    if BC == 'periodic':
        if not np.isclose(psi0[0], psi0[-1], rtol, atol):
            raise ValueError(f'Periodic bdry. cond. not satisfied with rtol={rtol:.2e}, atol={atol:.2e}.')
    elif BC == 'dirichlet':
        if not np.isclose(psi0[0], 0, rtol, atol):
            raise ValueError(f'Left bdry. cond. not satisfied with rtol={rtol:.2e}, atol={atol:.2e}.')
        if not np.isclose(psi0[-1], 0, rtol, atol):
            raise ValueError(f'Right bdry. cond. not satisfied with rtol={rtol:.2e}, atol={atol:.2e}.')
    elif BC == 'pml':
        pass
    else:
        raise ValueError('Boundaries must be "periodic", "dirichlet" or "pml".')


def check_bdry(t, x1, x2):
    '''Checks whether boundaries are valid or not.'''

    L = x2(t) - x1(t)
    cond = L < 0
    if np.any(cond):
        i = np.argmax(cond)
        raise ValueError(f'Boundary definition fails for t={t[i]:.2f}, negative domain length.')


def coords(x1, x2, Nx, BC):
    '''Meshes for x and its derivatives.'''

    @jit
    def dx(t):
        return (x2(t) - x1(t)) / (Nx - 1)
    if BC == 'pml':
        N = jnp.arange(-int(D*Nx), Nx + int(D*Nx))
    else:
        N = jnp.arange(Nx)
    @jit
    def x(t):
        return x1(t) + dx(t) * N
    @jit
    def x_t(t):
        x1_t = grad(x1)
        dx_t = grad(dx)
        return x1_t(t) + dx_t(t) * N
    @jit
    def x_tt(t):
        x1_t = grad(x1)
        x1_tt = grad(x1_t)
        dx_t  = grad(dx)
        dx_tt = grad(dx_t)
        return x1_tt(t) + dx_tt(t) * N

    return len(N), dx, x, x_t, x_tt


def extrapolate(psi, x, x1, x2, Nx):
    '''Smoothly tapers a wavefunction to zero in absorbing layers.'''

    x0 = np.linspace(x1, x2, Nx)
    x0 = np.concatenate(([x[0]], x0, [x[-1]]))
    amplitude = np.abs(psi)
    phase = np.angle(psi)
    cs = CubicSpline(x0, np.concatenate(([0], amplitude, [0])), bc_type='natural')
    amplitude_ext = cs(x)
    idx = int(D*Nx)
    phase_ext = np.concatenate(([phase[0]]*idx, phase, [phase[-1]]*idx))
    psi_ext = amplitude_ext * np.exp(1j*phase_ext)

    return psi_ext


@jit
def stretch(x, x1, x2):
    '''Complex coordinate stretching involved in PML.'''

    s1 = - (x1 - x)**3 * (x < x1)
    s2 = (x - x2)**3 * (x > x2)
    sigma = S0 * (s1 + s2)

    return x + jnp.exp(1j * G) * sigma


def compute_Veff(x, x_t, x_tt, x1, x2, V, BC):
    '''Effective potential in moving domain.'''

    if BC == 'pml':
        @jit
        def Veff(t):
            F   = stretch
            F_x = vmap(grad(F, holomorphic=True), (0, None, None))
            F   = F(x(t), x1(t), x2(t))
            F_x = F_x(x(t)+0j, x1(t), x2(t))
            Vbc = F * x_tt(t) + (F_x - .5) * x_t(t)**2
            return V(t, x(t)) + Vbc
        return Veff
    @jit
    def Veff(t):
        Vbc = x(t) * x_tt(t) + .5 * x_t(t)**2
        return V(t, x(t)) + Vbc
    return Veff
        

def compute_expT(x, x1, x2, Nx, BC):
    '''Kinetic energy time step.'''

    if BC == 'periodic':
        def expT(psi, t, dt, dx):
            k = 2 * np.pi * fft.fftfreq(Nx, dx)
            psi_f = fft.fft(psi)
            psi_f = psi_f * np.exp(- .5j * dt * k**2)
            psi = fft.ifft(psi_f)
            return psi
        return expT
    if BC == 'dirichlet':
        def expT(psi, t, dt, dx):
            k = np.pi * np.arange(1, Nx + 1) / (Nx * dx)
            psi_f = fft.dst(psi, type=1)
            psi_f = psi_f * np.exp(- .5j * dt * k**2)
            psi = fft.idst(psi_f, type=1)
            return psi
        return expT
    if BC == 'pml':
        @jit
        def F(t):
            return stretch(x(t), x1(t), x2(t))
        @jit
        def F_x(t):
            F_x = vmap(grad(stretch, holomorphic=True), (0, None, None))
            return F_x(x(t)+0j, x1(t), x2(t))
        def expT(psi, t, dt, dx):
            k = 2 * np.pi * fft.fftfreq(Nx, dx)
            psi_f = fft.fft(F_x(t) * psi)
            psi_f = psi_f * np.exp(- .5j * dt * k**2)
            psi = fft.ifft(psi_f)
            return psi
        return expT


@jit
def cutoff_periodic(psi_f, k, fraction=F):
    '''Frequency noise threshold for periodic BC.'''

    n = len(k)

    k_neg   = k[-1:n//2+n%2-1:-1]
    f_neg   = psi_f[-1:n//2+n%2-1:-1]
    pow_neg = jnp.abs(f_neg)**2
    cum_neg = jnp.cumsum(pow_neg) / jnp.sum(pow_neg)
    i = jnp.argmax(cum_neg >= fraction)
    kmin = k_neg[i]

    k_pos   = k[:n//2+n%2]
    f_pos   = psi_f[:n//2+n%2]
    pow_pos = jnp.abs(f_pos)**2
    cum_pos = jnp.cumsum(pow_pos) / jnp.sum(pow_pos)
    i = jnp.argmax(cum_pos >= fraction)
    kmax = k_pos[i]

    return kmin, kmax


@jit
def cutoff_dirichlet(psi_f, k, fraction=F):
    '''Frequency noise threshold for dirichlet BC.'''

    power = jnp.abs(psi_f)**2
    cumulative_power = jnp.cumsum(power) / jnp.sum(power)
    i = jnp.argmax(cumulative_power >= fraction)

    return k[i]


def energy(psi, x, V, BC):
    '''Average energy and standard deviation.'''

    Nx = len(x)
    dx = np.diff(x)[0]
    if BC == 'periodic':
        k = 2 * np.pi * fft.fftfreq(Nx, dx)
        psi_f = fft.fft(psi)
        kmin, kmax = cutoff_periodic(psi_f, k)
        k_filter = (k >= kmin) * (k <= kmax)
        T_psi_f = k**2/2 * psi_f * k_filter
        T_psi = fft.ifft(T_psi_f)
        H_psi = T_psi + V * psi
    elif BC == 'dirichlet':
        k = np.pi * np.arange(1, Nx + 1) / (Nx * dx)
        psi_f = fft.dst(psi, type=1)
        kc = cutoff_dirichlet(psi_f, k)
        k_filter = k <= kc
        T_psi_f = k**2/2 * psi_f * k_filter
        T_psi = fft.idst(T_psi_f, type=1)
        H_psi = T_psi + V * psi
    E  = np.trapezoid(psi.conj()*H_psi, x).real
    E2 = np.trapezoid(np.abs(H_psi)**2, x).real
    dE = np.sqrt(max(0, E2 - E**2))

    return E, dE
