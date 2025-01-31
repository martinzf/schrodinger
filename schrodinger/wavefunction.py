"""
wavefunction.py: Module containing Wavefunction object.
"""

from dataclasses import dataclass
from collections.abc import Callable

from jax import vmap
import numpy as np

from . import visuals
from . import utils

@dataclass
class Wavefunction:
    '''Class to represent a wavefunction on a moving domain.'''

    amplitude: np.ndarray
    t:         np.ndarray
    x1:        Callable[[float], float]
    x2:        Callable[[float], float]
    V:         Callable[[float, np.ndarray], np.ndarray]
    closed:    bool = True

    def __post_init__(self):
        x = self._x(self.t)
        V = self.V(self.t, x)
        if self.closed:
            self._normalize()
            self.E, self.dE = utils.energy(self.amplitude, x, V)

    def _x(self, t):
        Nx = self.amplitude.shape[1]
        return np.linspace(vmap(self.x1)(t), vmap(self.x2)(t), Nx).T

    def _normalize(self):
        norm_2 = np.trapezoid(np.abs(self.amplitude)**2, self._x(self.t), axis=1)
        self.amplitude = self.amplitude / np.sqrt(norm_2)[:, np.newaxis]

    def plot(self, t=None, x=None):
        if (t is not None) and (x is not None):
            raise ValueError("Provide either 't' or 'x', but not both.")

        Nt, Nx = self.amplitude.shape
        if t is not None:
            if t < self.t[0] or t > self.t[-1]:
                raise ValueError(f"'t' value outside of range [{self.t[0]:.2f}, {self.t[-1]:.2f}]")
            i = np.argmin(np.abs(self.t - t))
            x = np.linspace(self.x1(self.t[i]), self.x2(self.t[i]), Nx)
            if self.closed:
                title = rf't={self.t[i]:.2f}, E={self.E[i]:.2f}$\pm${self.dE[i]:.2f}'
            else:
                pass
            visuals.plot(x, self.amplitude[i], xlabel='x', title=title)
        elif x is not None:
            x1, x2 = self.x1(self.t), self.x2(self.t)
            xmin, xmax = np.min(x1), np.max(x2)
            if x < xmin or x > xmax:
                raise ValueError(f"'x' value outside of range [{xmin:.2f}, {xmax:.2f}]")
            mask = np.where((x1 < x) * (x < x2), 1, np.nan)
            def x_fun(t):
                return np.linspace(self.x1(t), self.x2(t), Nx)
            i = np.argmin(np.abs(x_fun(self.t) - x), axis=0)
            title = f'x={x:.2f}'
            visuals.plot(self.t, self.amplitude[list(range(Nt)),i]*mask, xlabel='t', title=title)
        else:
            Nx = self.amplitude.shape[1]
            ratio_x, t_ds, psi_ds = visuals.downsample(self.amplitude, self.t)
            T = np.repeat(t_ds, psi_ds.shape[1])
            X = self._x(t_ds)[:, ::ratio_x].ravel()
            PSI = psi_ds.ravel()
            visuals.trisurf(T, X, PSI)

    def animate(self):
        return visuals.animate_wf(self)
