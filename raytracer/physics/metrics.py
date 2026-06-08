"""Metric components and Christoffel contractions for Schwarzschild spacetime.

This module computes the relativistic geodesic derivatives for positions and
momenta using hardware-agnostic array primitives.
"""

"""Metric components and Christoffel contractions for Schwarzschild spacetime."""

from typing import Tuple
import cupy as cp
import numpy as np


def get_schwarzschild_derivatives(
    pos: np.ndarray, mom: np.ndarray, mass: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the geodesic derivatives dx^\mu/d\lambda and dp^\mu/d\lambda."""
    xp = cp.get_array_module(pos)

    r = pos[:, 1]
    # CLIP THETA: Prevents division by zero at the mathematical poles (0 and pi)
    theta = xp.clip(pos[:, 2], 1e-5, xp.pi - 1e-5)

    p_t = mom[:, 0]
    p_r = mom[:, 1]
    p_th = mom[:, 2]
    p_ph = mom[:, 3]

    rs = 2.0 * mass
    f = 1.0 - (rs / r)

    dx_dlambda = mom
    dp_dlambda = xp.zeros_like(mom)

    # Time component
    dp_dlambda[:, 0] = -(rs / (r * r * f)) * p_t * p_r

    # Radial component (RESTORED * f in the first term!)
    dp_dlambda[:, 1] = (
        -(rs * f / (2.0 * r * r)) * (p_t**2)
        + (rs / (2.0 * r * r * f)) * (p_r**2)
        + r * f * (p_th**2)
        + r * f * (xp.sin(theta) ** 2) * (p_ph**2)
    )

    # Theta component
    dp_dlambda[:, 2] = -(2.0 / r) * p_r * p_th + xp.sin(theta) * xp.cos(
        theta
    ) * (p_ph**2)

    # Phi component
    dp_dlambda[:, 3] = (
        -(2.0 / r) * p_r * p_ph - (2.0 / xp.tan(theta)) * p_th * p_ph
    )

    return dx_dlambda, dp_dlambda
