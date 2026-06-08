"""Central orchestrator for the GPU-Accelerated Relativistic Ray Tracer.

Coordinates configuration parsing, initial condition generation, parallelized
geodesic integration, and image file compilation.
"""

import os
import time

import cupy as cp
import matplotlib.pyplot as plt
from raytracer.camera import generate_initial_conditions
from raytracer.physics.integrators import integrate_geodesics
from raytracer.utils.io_helpers import load_config
from raytracer.visualization.render import render_frame


def main() -> None:
    """Executes the pipeline sequentially and exports the rendering asset."""
    print("=" * 60)
    print("      INITIALIZING RELATIVISTIC RAY TRACING PIPELINE      ")
    print("=" * 60)

    # 1. Load configuration parameters
    config_path = "configs/config.yaml"
    config = load_config(config_path)
    print(f"[+] Configuration successfully loaded from: {config_path}")

    # 2. Build camera viewport parameters and project initial conditions
    print("[+] Generating 3D initial positions and 4-momenta on GPU...")
    start_time = time.time()
    init_pos, init_mom = generate_initial_conditions(config)

    # 3. Fire the parallelized geodesic solver loop
    print(f"[+] Integrating trajectories for {init_pos.shape[0]} rays...")
    final_pos, final_mom, disk_hit_radii = integrate_geodesics(
        init_pos, init_mom, config
    )

    # 4. Map final coordinates to the lensed color space
    print("[+] Processing pixel values and texture mapping...")
    gpu_image = render_frame(final_pos, disk_hit_radii, config)

    # 5. Offload array from GPU VRAM to Host System RAM (numpy conversion)
    print("[+] Transferring image data from VRAM to Host CPU...")
    cpu_image = cp.asnumpy(gpu_image)

    execution_time = time.time() - start_time
    print(
        f"[✓] Pipeline complete! Core compute time: {execution_time:.4f} seconds"
    )

    # 6. Save the final image output asset to disk
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "black_hole_lensing.png")

    # Save using matplotlib without borders or axes
    plt.imsave(output_path, cpu_image)
    print(f"[✓] High-fidelity asset successfully saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
