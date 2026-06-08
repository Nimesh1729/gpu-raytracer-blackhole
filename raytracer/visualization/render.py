"""Vectorized post-processing engine for accretion disk visualizations."""

from typing import Any, Dict
import cupy as cp


def render_frame(
    final_pos: cp.ndarray, disk_hit_radii: cp.ndarray, config: Dict[str, Any]
) -> cp.ndarray:
    """Renders scenes containing both an accretion disk and a background grid."""
    width = config["camera"]["width"]
    height = config["camera"]["height"]
    r_inner = config["simulation"]["r_inner"]

    num_pixels = final_pos.shape[0]
    image_flat = cp.zeros((num_pixels, 3), dtype=cp.float32)

    # 1. Mask out what hit the disk vs what reached deep space
    r_final = final_pos[:, 1]
    hit_disk_mask = disk_hit_radii > 0.0
    escaped_mask = (r_final > r_inner) & (~hit_disk_mask)

    # 2. Render the glowing accretion disk plasma
    if cp.any(hit_disk_mask):
        r_hit = disk_hit_radii[hit_disk_mask]

        # Calculate a thermal temperature gradient based on radius.
        # Matter closer to the black hole is hotter and glows brighter!
        intensity = 4.0 / (r_hit - 2.5)  # Spikes near the inner rim
        intensity = cp.clip(intensity, 0.0, 1.0)[:, cp.newaxis]

        # Professional hot gas color profile: Core orange fading to deep red
        color_hot = cp.array([1.0, 0.6, 0.1], dtype=cp.float32)  # Bright Orange
        color_warm = cp.array([0.5, 0.05, 0.0], dtype=cp.float32)  # Dark Red

        image_flat[hit_disk_mask] = (intensity * color_hot) + (
            (1.0 - intensity) * color_warm
        )

    # 3. Render the background celestial grid for escaping rays
    if cp.any(escaped_mask):
        theta = final_pos[escaped_mask, 2]
        phi = final_pos[escaped_mask, 3]
        phi_wrapped = cp.mod(phi, 2.0 * cp.pi)

        grid_frequency = 24.0
        pattern_theta = cp.sin(grid_frequency * (theta + 0.5))
        pattern_phi = cp.sin(grid_frequency * phi_wrapped)
        checker = cp.sign(pattern_theta * pattern_phi)

        color_grid = cp.array(
            [0.05, 0.1, 0.2], dtype=cp.float32
        )  # Dim background grid
        color_space = cp.array(
            [0.0, 0.0, 0.0], dtype=cp.float32
        )  # Pitch black space

        selector = ((checker + 1.0) / 2.0)[:, cp.newaxis]
        image_flat[escaped_mask] = (selector * color_grid) + (
            (1.0 - selector) * color_space
        )

    # 4. Reshape back to image dimensions
    image_2d = image_flat.reshape((height, width, 3))
    return image_2d
