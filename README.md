# Relativistic Ray Tracer: Schwarzschild Spacetime and Accretion Disk Dynamics

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![CUDA/CuPy](https://img.shields.io/badge/backend-CuPy%20%2F%20NumPy-73B504.svg)
![Integration](https://img.shields.io/badge/integrator-RK4-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A GPU-accelerated general relativistic ray tracer for simulating photon trajectories in Schwarzschild spacetime. The project numerically integrates null geodesics around a non-rotating black hole and renders gravitational lensing effects, black hole shadows, photon rings, and equatorial accretion disk intersections.

The solver is implemented from scratch in Python and uses a vectorized fourth-order Runge-Kutta integrator. It supports both CuPy for GPU execution and NumPy for CPU fallback, allowing the same physics pipeline to run across different hardware backends.

---

## Key Features

- **General relativistic geodesic integration:** Directly evaluates the coupled geodesic equations using analytical Christoffel symbols, expressed in spherical coordinates `(t, r, theta, phi)`.

- **GPU-accelerated computation:** Uses CuPy to evolve large batches of photon trajectories in parallel on NVIDIA GPUs, while retaining NumPy compatibility for CPU-based runs.

- **Stable coordinate handling:** Applies numerical safeguards near polar coordinate singularities and normalizes azimuthal coordinates using `phi mod 2pi`, improving stability during long integrations.

- **Accretion disk intersection tracking:** Detects photon crossings through the equatorial plane and determines whether rays intersect a finite disk region around the black hole.

- **Celestial grid lensing:** Maps escaping photon trajectories to a background celestial grid, making the distortion caused by strong-field gravitational lensing visually clear.

- **Automated physics tests:** Includes `pytest`-based checks for conserved quantities, including the time-translation invariant associated with photon energy, to monitor numerical drift during integration.

---

## Mathematical Framework

Photon trajectories are modeled as null geodesics in Schwarzschild spacetime. The second-order geodesic equations are rewritten as a first-order system by evolving both the photon position `x^mu` and four-momentum `p^mu`:

```text
dx^mu / dlambda = p^mu
```

```text
dp^mu / dlambda = -Gamma^mu_{alpha beta} p^alpha p^beta
```

Here, `lambda` is an affine parameter and `Gamma^mu_{alpha beta}` are the Christoffel symbols of the Schwarzschild metric.

The integration is performed backward from the camera into the scene. At each step, rays are classified according to whether they escape to the celestial sphere, fall into the event horizon, or intersect the accretion disk.

---

## Rendering Modes

### 1. Equatorial Accretion Disk Rendering

This mode simulates an opaque, geometrically thin accretion disk in the equatorial plane.

The renderer:

- Tracks sign changes in `theta - pi/2` to detect equatorial crossings.
- Checks whether crossing points lie within the disk bounds.
- Applies a radial intensity profile to approximate hotter inner disk regions.
- Resolves primary disk images and higher-order lensed structures, including photon-ring contributions near `r = 3M`.

The current disk model uses a simplified emissivity profile and is intended as a physically motivated visualization rather than a full radiative-transfer simulation.

### 2. Celestial Grid Lensing

This mode maps escaping rays onto an analytical background sphere. The resulting image shows how Schwarzschild spacetime bends light from different viewing directions.

It is useful for visualizing:

- Black hole shadow formation.
- Strong gravitational lensing.
- Coordinate distortion near the photon sphere.
- The boundary between escaping and captured photon trajectories.

---

## Installation

This project uses a modern `pyproject.toml` configuration and supports editable installation for development.

### Prerequisites

- Python 3.9+
- Optional: NVIDIA GPU with CUDA-compatible drivers
- CuPy for GPU acceleration
- NumPy fallback for CPU execution

### Setup

```bash
git clone https://github.com/yourusername/relativistic-ray-tracer.git
cd relativistic-ray-tracer

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -e .[dev]
```

---

## Usage

To run the full simulation and generate a high-fidelity render:

```bash
python main.py
```

Generated images are saved to the `outputs/` directory.

---

## Physics Validation

To run the automated physics validation suite:

```bash
python -m pytest
```

The tests verify conserved quantities and help ensure that the numerical integration remains physically consistent within the expected tolerance.

---

## Configuration

All physical parameters, hardware backends, integration settings, and camera definitions are controlled through:

```text
configs/config.yaml
```

Important configuration options include:

- `step_size`: Controls the integration step length. Smaller values generally improve numerical precision but increase runtime.
- `max_steps`: Sets the maximum number of integration steps per ray.
- `backend`: Selects the numerical backend. Use `cupy` for GPU acceleration or `numpy` for CPU fallback.
- Camera parameters: Define the observer position, viewing direction, field of view, and image resolution.

---

## Project Structure

```text
relativistic_ray_tracer/
├── configs/
│   └── config.yaml          # Global simulation parameters
├── raytracer/
│   ├── physics/
│   │   ├── metrics.py       # Spacetime derivatives and Christoffel contractions
│   │   └── integrators.py   # RK4 parallel marching and disk intersection logic
│   ├── utils/
│   │   └── io_helpers.py    # Memory and asset management
│   └── visualization/
│       ├── camera.py        # Camera initialization and ray setup
│       └── render.py        # Post-processing, color mapping, and image shaping
├── tests/
│   └── test_physics.py      # Automated invariant verification tests
├── outputs/                 # Generated rendered images
├── main.py                  # Primary execution entry point
└── pyproject.toml           # Build system and dependency management
```

---

## Future Architecture

This computational baseline is designed to support two possible advanced research extensions.

### 1. Physics-Informed Neural Networks

A future extension may explore replacing or accelerating the explicit RK4 integration loop with a physics-informed neural network. Instead of marching each photon step by step, the network would learn a direct mapping from initial camera-ray conditions to final ray outcomes.

The Schwarzschild geodesic equations could be embedded into the training objective as a physics-based loss term. This would allow the model to learn trajectories that remain constrained by the underlying differential equations, while potentially enabling faster inference after training.

This phase would investigate whether a learned surrogate model can approximate ray evolution accurately enough for real-time or near-real-time rendering.

### 2. Solar Plasma Wave Propagation

The mathematical and computational structure of this project may also be extended toward plasma physics simulations. In particular, the tensor-based physics pipeline and high-performance numerical backend could be adapted to model localized wave propagation in magnetized solar plasma.

Possible targets include:

- Acoustic waves in solar plasma.
- Alfven waves guided by magnetic field lines.
- Localized magnetohydrodynamic wave propagation in the solar corona.

This would require moving beyond Schwarzschild geodesics into fluid and magnetohydrodynamic equations, but the existing project provides a useful foundation in numerical physics, vectorized simulation, and scientific visualization.

---
