"""Vectorized RK4 integration engine with active accretion disk intersection."""

from typing import Any, Dict, Tuple
import cupy as cp
import numpy as np
from raytracer.physics.metrics import get_schwarzschild_derivatives


def rk4_step(
    pos: cp.ndarray, mom: cp.ndarray, h: float, mass: float
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Advances position and momentum arrays by a single vectorized RK4 step."""
    k1_pos, k1_mom = get_schwarzschild_derivatives(pos, mom, mass)

    p2 = pos + 0.5 * h * k1_pos
    m2 = mom + 0.5 * h * k1_mom
    k2_pos, k2_mom = get_schwarzschild_derivatives(p2, m2, mass)

    p3 = pos + 0.5 * h * k2_pos
    m3 = mom + 0.5 * h * k2_mom
    k3_pos, k3_mom = get_schwarzschild_derivatives(p3, m3, mass)

    p4 = pos + h * k3_pos
    m4 = mom + h * k3_mom
    k4_pos, k4_mom = get_schwarzschild_derivatives(p4, m4, mass)

    next_pos = pos + (h / 6.0) * (k1_pos + 2.0 * k2_pos + 2.0 * k3_pos + k4_pos)
    next_mom = mom + (h / 6.0) * (k1_mom + 2.0 * k2_mom + 2.0 * k3_mom + k4_mom)

    return next_pos, next_mom


def integrate_geodesics(
    pos: np.ndarray, mom: np.ndarray, config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrates global parallel loops while tracking equatorial disk collisions.

    Returns:
        Tuple containing:
            - final_pos: Final coordinates (N, 4)
            - final_mom: Final momenta (N, 4)
            - disk_hit_radii: Array of shape (N,) containing the intersection radius
              if the ray hit the disk, or 0.0 if it missed.
    """
    xp = cp.get_array_module(pos)

    # Move data onto the GPU if using CuPy
    if config["simulation"]["backend"] == "cupy":
        pos_gpu = cp.asarray(pos)
        mom_gpu = cp.asarray(mom)
    else:
        pos_gpu = pos
        mom_gpu = mom

    h = config["simulation"]["step_size"]
    max_steps = config["simulation"]["max_steps"]
    r_inner = config["simulation"]["r_inner"]
    r_outer = config["simulation"]["r_outer"]
    mass = config["physics"]["mass"]

    num_rays = pos_gpu.shape[0]

    # Track which rays hit the disk and exactly where (radial coordinate r)
    disk_hit_radii = xp.zeros(num_rays, dtype=xp.float32)

    # Define disk profile limits (from just outside horizon to stable outer orbits)
    r_disk_min = 4.0 * mass
    r_disk_max = 12.0 * mass

    # Active mask: ray is running if it hasn't escaped, fallen in, or hit the disk
    active_mask = xp.ones(num_rays, dtype=xp.bool_)

    for _ in range(max_steps):
        if not xp.any(active_mask):
            break

        # Cache positions before the step to detect equatorial plane crossings
        old_theta = pos_gpu[:, 2] - (xp.pi / 2.0)

        # Advance active rays
        next_pos, next_mom = rk4_step(
            pos_gpu[active_mask], mom_gpu[active_mask], h, mass
        )

        pos_gpu[active_mask] = next_pos
        mom_gpu[active_mask] = next_mom

        # Check current conditions for active rays
        r_current = pos_gpu[:, 1]
        new_theta = pos_gpu[:, 2] - (xp.pi / 2.0)

        # EQUATORIAL INTERSECTION PHYSICS:
        # A change in sign of (theta - pi/2) means the ray sliced through the equator
        crossed_equator = (
            xp.sign(old_theta) != xp.sign(new_theta)
        ) & active_mask

        if xp.any(crossed_equator):
            # Verify if the crossing happened within the disk's physical boundaries
            hit_disk_now = (
                crossed_equator
                & (r_current >= r_disk_min)
                & (r_current <= r_disk_max)
            )
            if xp.any(hit_disk_now):
                # Record the radial impact position for the renderer
                disk_hit_radii[hit_disk_now] = r_current[hit_disk_now]
                # Freeze those rays permanently!
                active_mask[hit_disk_now] = False

        # General boundary conditions
        escaped = (r_current >= r_outer) & active_mask
        captured = (r_current <= r_inner) & active_mask

        active_mask[escaped] = False
        active_mask[captured] = False

    return pos_gpu, mom_gpu, disk_hit_radii
