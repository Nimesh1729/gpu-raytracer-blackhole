"""Virtual camera initialization for relativistic ray tracing.

This module maps a 2D flat-screen pixel grid into 3D initial positions
and 4-momenta in Schwarzschild spacetime using an orthonormal tetrad.
"""

from typing import Any, Dict, Tuple

import cupy as cp


def generate_initial_conditions(
    config: Dict[str, Any],
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Generates starting 4-positions and 4-momenta for all screen pixels.

    Places the observer on the equatorial plane (theta = pi/2) at a radial
    distance r_cam, looking directly toward the black hole origin.

    Args:
        config: The parsed configuration dictionary containing 'camera' and
            'physics' parameters.

    Returns:
        A tuple containing two CuPy arrays:
            - pos: Array of shape (N, 4) for positions (t, r, theta, phi).
            - mom: Array of shape (N, 4) for momenta (p_t, p_r, p_theta, p_phi).
    """
    # Extract parameters from the parsed configuration
    width = config["camera"]["width"]
    height = config["camera"]["height"]
    fov_deg = config["camera"]["fov_deg"]
    r_cam = config["camera"]["r_cam"]
    mass = config["physics"]["mass"]

    num_pixels = width * height
    aspect_ratio = width / height

    # 1. Construct the flat virtual screen grid (X, Y)
    fov_rad = cp.radians(fov_deg)
    x_max = cp.tan(fov_rad / 2.0)
    y_max = x_max / aspect_ratio

    # Use float32 to halve VRAM usage compared to standard float64
    x = cp.linspace(-x_max, x_max, width, dtype=cp.float32)
    y = cp.linspace(
        y_max, -y_max, height, dtype=cp.float32
    )  # Y inverted for graphics
    X, Y = cp.meshgrid(x, y)

    # Flatten arrays for contiguous memory allocation on the GPU
    X = X.ravel()
    Y = Y.ravel()

    # 2. Define initial positions in global Schwarzschild coordinates
    pos = cp.zeros((num_pixels, 4), dtype=cp.float32)
    pos[:, 0] = 0.0  # t = 0
    pos[:, 1] = r_cam  # r = observer distance
    pos[:, 2] = cp.pi / 2.0  # theta = equatorial plane
    pos[:, 3] = 0.0  # phi = 0

    # 3. Define photon momenta in the local orthonormal tetrad
    # Photons travel inward from the screen toward the black hole
    p_hat_r = -1.0 / cp.sqrt(1.0 + X**2 + Y**2)
    p_hat_t = cp.ones(num_pixels, dtype=cp.float32)
    p_hat_theta = Y * p_hat_r
    p_hat_phi = X * p_hat_r

    # 4. Transform local tetrad momenta to global Schwarzschild momenta
    rs = 2.0 * mass
    f = 1.0 - rs / r_cam

    mom = cp.zeros((num_pixels, 4), dtype=cp.float32)
    mom[:, 0] = p_hat_t / cp.sqrt(f)
    mom[:, 1] = p_hat_r * cp.sqrt(f)
    mom[:, 2] = p_hat_theta / r_cam
    mom[:, 3] = p_hat_phi / (r_cam * cp.sin(pos[:, 2]))

    return pos, mom
