# Research Journal and Development Log

**Project:** Relativistic Ray Tracer: Schwarzschild Spacetime and Accretion Disk Dynamics
**Context:** Computational physics project focused on general relativistic ray tracing

---

## Abstract and Initial Objectives

The objective of this project was to build a general relativistic ray tracer capable of simulating the bending of light around a non-rotating black hole. Instead of using standard 3D rendering approximations, the project directly integrates photon trajectories as null geodesics in Schwarzschild spacetime.

The initial goals were:

- Implement the Schwarzschild metric in geometric units.
- Derive and numerically integrate the geodesic equations.
- Trace photon paths backward from a virtual camera.
- Classify rays as escaping, captured, or disk-intersecting.
- Render gravitational lensing, the black hole shadow, and accretion disk structures.
- Validate the numerical solver using conserved physical quantities.

The project was designed as both a physics simulation and a numerical computing exercise, emphasizing correctness, modularity, and GPU acceleration.

---

## Phase 1: Mathematical Formulation

**Objective:** Translate the equations of general relativity into a computable numerical system.

The simulation begins with the Schwarzschild metric in geometric units, where `G = c = 1`:

$$
ds^2 =
-\left(1 - \frac{2M}{r}\right)dt^2
+ \left(1 - \frac{2M}{r}\right)^{-1}dr^2
+ r^2 d\theta^2
+ r^2 \sin^2\theta d\phi^2
$$

Since photons follow null geodesics, their spacetime interval satisfies:

$$
ds^2 = 0
$$

To make the equations suitable for numerical integration, the second-order geodesic equations were rewritten as a first-order system. Each photon is represented by its four-position `x^mu` and four-momentum `p^mu`:

$$
\frac{dx^\mu}{d\lambda} = p^\mu
$$

$$
\frac{dp^\mu}{d\lambda}
=
-\Gamma^\mu_{\alpha\beta}p^\alpha p^\beta
$$

Here, `lambda` is an affine parameter and `Gamma^mu_{alpha beta}` are the Christoffel symbols associated with the Schwarzschild metric.

The main implementation challenge in this phase was converting the tensor equations into stable, vectorized array operations that could be evaluated efficiently for many rays at once.

### Validation Strategy

To check whether the mathematical implementation was consistent, I added automated tests based on conserved quantities.

Because the Schwarzschild metric is time-independent, photon energy is conserved along a geodesic. Numerically, this corresponds to monitoring the covariant time component of momentum, `p_t`, and checking that it does not drift significantly during integration.

A `pytest` validation suite was added to measure relative drift in this quantity and confirm that the numerical error remains within an acceptable tolerance.

---

## Phase 2: Numerical Integration and Hardware Acceleration

**Objective:** Scale the geodesic solver so that many photon trajectories can be evolved efficiently.

The ray tracer uses a fourth-order Runge-Kutta integrator. RK4 was chosen because simple methods such as Euler integration accumulate too much error over long, curved trajectories. RK4 evaluates the derivative field several times per step, giving a better balance between accuracy and implementation simplicity.

The integration loop evolves each ray through Schwarzschild spacetime while checking whether it:

- Escapes to the background celestial sphere.
- Approaches the black hole capture region.
- Crosses the equatorial accretion disk.
- Reaches the maximum integration step count.

To make the solver practical at image-rendering resolutions, the implementation avoids Python loops over individual rays. Instead, photon states are stored as arrays and evolved in parallel.

The backend can use:

- `CuPy` for GPU-accelerated array computation on NVIDIA hardware.
- `NumPy` as a CPU fallback backend.

This design allows the same simulation code to run on different hardware while preserving the same mathematical pipeline.

---

## Phase 3: Numerical Stability and Coordinate Issues

**Objective:** Identify and resolve numerical problems caused by strong-field gravity and spherical coordinates.

### 1. Near-Horizon Precision Instability

An early implementation used an inner stopping threshold very close to the event horizon, around `r = 2.001M`. Near this radius, the Schwarzschild factor

$$
f(r) = 1 - \frac{2M}{r}
$$

becomes very small. As a result, expressions involving division by `f(r)` become numerically unstable, especially when using `float32` arithmetic on the GPU.

This caused some rays to behave incorrectly near the black hole, producing unstable trajectories and visual artifacts.

**Resolution:**
The simulation was adjusted to use a more conservative inner termination region. This avoids integrating too close to the coordinate singularity at the Schwarzschild radius, where the chosen coordinates become numerically difficult to work with.

This is a practical rendering decision rather than a claim that the event horizon physically moves. The true Schwarzschild event horizon remains at `r = 2M`; the adjusted threshold is a numerical safeguard for stable rendering.

### 2. Spherical Coordinate Singularity

Another issue appeared near the polar axis, where `theta = 0` or `theta = pi`.

In spherical coordinates, some terms in the equations contain factors such as `sin(theta)` or `tan(theta)`. These become singular or poorly conditioned near the poles. In rendered images, this produced thin visual artifacts at the top and bottom of the black hole shadow.

**Resolution:**
A small coordinate clipping operation was added to keep `theta` away from the exact singular values:

$$
\theta \in [\epsilon, \pi - \epsilon]
$$

This preserves vectorized execution while preventing division-by-zero errors. The remaining small artifacts are understood as numerical signatures of the spherical coordinate system, not physical features of the black hole.

---

## Phase 4: Accretion Disk Rendering and Photon-Ring Structure

**Objective:** Extend the simulation beyond vacuum lensing and include an equatorial accretion disk.

The RK4 integration loop was extended to monitor crossings of the equatorial plane:

$$
\theta = \frac{\pi}{2}
$$

If a ray crossed the equatorial plane between two integration steps, the code checked whether the crossing radius fell inside the disk bounds. Rays satisfying this condition were classified as disk hits.

The disk was modeled as a geometrically thin, opaque disk with a simple radial intensity profile. The implementation used disk bounds such as:

$$
4M \le r \le 12M
$$

This range avoids the innermost unstable region while still placing bright material close enough to the black hole to show strong lensing effects.

### Visual Results

The renderer produced several expected qualitative features:

- A primary accretion disk image.
- Strong bending of the disk around the black hole.
- A dark central shadow region.
- Higher-order lensed structures near the photon sphere.
- Thin ring-like features associated with photons passing close to `r = 3M`.

A simplified thermal intensity model was used, with brightness increasing toward smaller radii. This gives a physically motivated visualization of a hotter inner disk, although it is not a full radiative-transfer model.

The photon ring was treated carefully in the interpretation. The rendered ring-like structure is consistent with strong lensing near the photon sphere, but a full quantitative analysis would require higher-resolution convergence tests and more detailed radiative modeling.

---

## Phase 5: Testing and Validation

**Objective:** Add automated checks to ensure that the simulation remains physically and numerically consistent.

The validation suite focuses on conserved quantities and integration stability.

The most important test tracks the conserved photon energy associated with time-translation symmetry in Schwarzschild spacetime. Since the metric is static, this quantity should remain approximately constant along each ray.

The test suite checks:

- Whether the solver runs without numerical exceptions.
- Whether conserved quantities remain stable within tolerance.
- Whether the integration backend behaves consistently.
- Whether changes to the physics code introduce unexpected drift.

These tests do not prove that the entire renderer is physically complete, but they provide an important guardrail against implementation errors.

---

## Phase 6: Future Research Directions

The current project establishes a working numerical baseline for relativistic ray tracing. Two possible future extensions are especially relevant.

### 1. Physics-Informed Neural Networks

Explicit RK4 integration is accurate but computationally expensive. A future extension could train a physics-informed neural network to approximate the ray-tracing map directly.

Instead of evolving every photon step by step, the model would learn a mapping from initial camera-ray conditions to final ray outcomes. The Schwarzschild geodesic equations could be included in the loss function so that the network is encouraged to respect the underlying physics.

This would turn the project into a surrogate-modeling problem:

- Inputs: initial ray position and direction.
- Outputs: escape direction, disk intersection, or capture classification.
- Constraint: geodesic residuals should remain small.

The main research question would be whether the learned model can approximate the RK4 solver accurately enough while providing faster inference.

### 2. Solar Plasma Wave Propagation

The project may also serve as a stepping stone toward plasma physics simulations.

Although black hole ray tracing and solar plasma dynamics involve different physical equations, both require:

- Numerical solution of differential equations.
- Careful treatment of coordinate systems.
- Stable integration over many grid points or trajectories.
- Efficient array-based computation.
- Scientific visualization of field evolution.

A future solar extension could focus on localized magnetohydrodynamic wave propagation in the solar corona, including acoustic and Alfvén waves.

This would require replacing the Schwarzschild geodesic equations with fluid or MHD equations, but the computational lessons from this project would still transfer: modular physics code, vectorized numerical kernels, backend abstraction, testing, and visualization.

---

## Summary

This project successfully developed a modular general relativistic ray tracer for Schwarzschild spacetime. The final system integrates null geodesics, supports GPU acceleration, renders gravitational lensing and accretion disk structures, and includes automated validation checks for conserved quantities.

The most important lessons from the project were:

- General relativity can be translated into a numerical simulation through the geodesic equations.
- Coordinate systems can introduce numerical artifacts that must be handled carefully.
- Conservation laws provide powerful tests for physics engines.
- GPU array programming is essential for scaling ray tracing to large numbers of photon paths.
- A careful distinction must be maintained between physically exact claims and simplified rendering models.

The project now provides a strong computational baseline for future work in relativistic visualization, physics-informed surrogate modeling, and broader numerical physics simulations.
