"""Verification suite for the relativistic ray tracer physics engine.

This module uses Pytest to run analytical and numerical invariant checks
on the Schwarzschild geodesic equations.
"""

import numpy as np
from raytracer.physics.integrators import integrate_geodesics
from raytracer.physics.metrics import get_schwarzschild_derivatives


def test_flat_space_photon_linearity():
    """Verifies that in flat space (M=0), photon momentum remains invariant."""
    # 1. Arrange: Create a flat-space mock scenario
    # Two parallel rays at (t=0, r=10, theta=pi/2, phi=0)
    pos = np.array([[0.0, 10.0, np.pi / 2, 0.0], [0.0, 10.0, np.pi / 2, 0.1]])
    # Photons moving directly inward radially (p_r = -1)
    mom = np.array([[1.0, -1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]])
    mass = 0.0  # Turn off gravity completely

    # 2. Act: Calculate derivatives
    _, dp_dlambda = get_schwarzschild_derivatives(pos, mom, mass)

    # 3. Assert: In flat space, acceleration must be exactly zero
    assert np.allclose(dp_dlambda, 0.0, atol=1e-6), (
        "Failing: Photon accelerated in flat Minkowski space!"
    )


def test_energy_conservation():
    """Verifies that the Killing invariant p_t is conserved along the geodesic."""
    # 1. Arrange: Create a standard simulation tracking profile
    mock_config = {
        "physics": {"mass": 1.0},
        "simulation": {
            "max_steps": 100,
            "r_inner": 2.001,
            "r_outer": 50.0,
            "step_size": 0.01,
            "backend": "numpy",  # <-- ADD THIS LINE (numpy is faster for unit tests)
        },
    }

    # Initialize a ray at r=15
    pos = np.array([[0.0, 15.0, np.pi / 2.0, 0.0]], dtype=np.float32)
    mom = np.array([[1.05, -0.5, 0.0, 0.1]], dtype=np.float32)

    # Calculate initial true lower-index energy: p_t = g_tt * p^t
    f_initial = 1.0 - (2.0 * mock_config["physics"]["mass"] / pos[:, 1])
    initial_energy = -f_initial * mom[:, 0]

    # 2. Act: Run the integration loop over 100 steps
    final_pos, final_mom, _ = integrate_geodesics(pos, mom, mock_config)

    # Calculate final true lower-index energy: p_t = g_tt * p^t
    f_final = 1.0 - (2.0 * mock_config["physics"]["mass"] / final_pos[:, 1])
    final_energy = -f_final * final_mom[:, 0]

    # 3. Assert: Check that the true conserved energy did not drift
    assert np.allclose(initial_energy, final_energy, rtol=1e-3), (
        f"Failing: Energy drifted! Initial: {initial_energy}, Final: {final_energy}"
    )
