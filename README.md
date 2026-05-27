<div align="center">

# Kolmogorov Flow Tracer & Streakline Simulation

### A pseudo-spectral solver for 2D Kolmogorov flow with tracer transport and stochastic particle streaklines

`Python` · `NumPy` · `SciPy` · `Matplotlib` · `Fourier spectral methods`

</div>

---

## Overview

This repository simulates **two-dimensional Kolmogorov flow** on a periodic square domain and visualizes the evolution of a passive tracer field together with stochastic source-injected particles.

The code evolves three coupled objects:

| Variable | Meaning | Representation |
|---|---|---|
| `W` | vorticity field | Fourier space |
| `C_hat` | passive tracer concentration | Fourier space |
| `xs` | streakline/source particles | physical particle coordinates |

The main model combines:

- a pseudo-spectral vorticity solver,
- a passive scalar tracer equation with source forcing,
- stochastic particle advection-diffusion,
- periodic boundary conditions,
- image output for making a movie.

---

## Mathematical Model

### Flow field

The flow is represented by the vorticity field `W`. The code evolves the vorticity using a Crank-Nicolson / Heun-type time step.

The physical velocity field is recovered from vorticity through the Fourier-space relation between vorticity and velocity.

```text
vorticity W  →  velocity u, v
```

The Kolmogorov forcing is applied in Fourier space through selected modes controlled by `n`.

---

### Tracer concentration

The tracer concentration satisfies an advection-diffusion-source equation of the form

```text
∂C/∂t + u ∂C/∂x + v ∂C/∂y = D ΔC + source
```

In the code, the tracer field is stored in Fourier space as `C_hat`. The function `C2f` computes the nonlinear advection term and adds the Gaussian source forcing.

The tracer source is a Gaussian centered at

```text
(x, y) = (π, π)
```

with amplitude `a` and width `sigma`.

---

### Particle streaklines with diffusion

The particles `xs` are stochastic Lagrangian markers. Each particle follows

```text
dX = u(X,t) dt + sqrt(2 D_particles) dB
```

where:

- `u(X,t)` is the velocity interpolated at the particle location,
- `D_particles` is the particle diffusion coefficient,
- `dB` is Brownian noise,
- periodic wrapping keeps particles inside the domain.

The particle update uses a Heun/trapezoidal drift step plus Langevin diffusion:

```text
drift = 0.5 * (velocity_old + velocity_new) * dt
diffusion = sqrt(2 * D_particles * dt) * random_normal
```

A new particle is injected at `(π, π)` every `inject_every` small solver steps.

---

## Code Structure

### `Evolve` class

The `Evolve` class contains the numerical solver and particle update methods.

#### Important methods

| Method | Purpose |
|---|---|
| `fft2(q)` | normalized 2D FFT |
| `ifft2(q_hat)` | inverse normalized 2D FFT |
| `init_C()` | initializes the tracer field |
| `W2u(W)` | converts Fourier vorticity to velocity |
| `u2w(u, v)` | converts velocity to vorticity |
| `forcing2f()` | builds the Gaussian tracer source |
| `C2f(C_hat, W, forcing_hat)` | computes tracer RHS in Fourier space |
| `W2f(W)` | computes vorticity RHS |
| `dWdt(W, T)` | evolves the flow only |
| `_periodic_interpolators(u, v)` | builds periodic velocity interpolators |
| `dCdt_CN_Heunn_with_particles(...)` | coupled update of `W`, `C_hat`, and `xs` |

---

## Main Simulation Loop

The main loop stores snapshots, then advances the coupled system:

```python
for frame in range(n_frames):
    Cs.append(KFlow.ifft2(C_hat))
    Ws.append(KFlow.ifft2(W))
    times.append(frame * dT)
    particle_history.append(copy.deepcopy(xs))

    C_hat, W, xs = KFlow.dCdt_CN_Heunn_with_particles(
        C_hat,
        W,
        forcing_hat,
        xs,
        dT,
        D_particles=D_particles,
        seed=particle_seed,
        inject_every=inject_every
    )
```

This means that during every output interval `dT`, the code internally performs many small `dt` updates. At each small step:

1. the vorticity field `W` is updated,
2. the tracer field `C_hat` is updated,
3. all particles `xs` are advected and diffused,
4. a new source particle may be injected.

---

## Parameters

Typical parameters used in the script:

| Parameter | Description | Example |
|---|---|---|
| `Nx`, `Ny` | grid resolution | `128, 128` |
| `dt` | solver time step | `0.001` |
| `Re` | Reynolds number | `14.4` |
| `n` | Kolmogorov forcing mode | `2` |
| `D` | tracer diffusion coefficient | `1e-3` |
| `n_frames` | number of saved frames | `1000` |
| `dT` | physical time between saved frames | `0.25` |
| `D_particles` | particle diffusion coefficient | `D` |
| `inject_every` | inject a particle every N small steps | `10` |

---

## Required Input File

The script loads a precomputed spun-up flow field:

```python
W = pickle.load(open('Wini.p', 'rb'))
```

Place `Wini.p` in the same directory as the script before running.

If `Wini.p` does not exist, you can generate it by uncommenting or adding a spin-up block such as:

```python
W = KFlow.dWdt(W, spinup_steps * dt)
pickle.dump(W, open('Wini.p', 'wb'))
```

For reproducible comparisons, do **not** overwrite `Wini.p` unless you intentionally want a new initial flow field.

---

## Output

The script saves images to the `Video/` directory:

```text
Video/0001.png
Video/0002.png
Video/0003.png
...
```

Each frame shows:

- the normalized tracer concentration field,
- red stochastic particles,
- velocity vectors sampled on a coarser grid.

---

## Installation

Create a Python environment with the required packages:

```bash
pip install numpy scipy matplotlib
```

---

## Running the Simulation

From the repository directory:

```bash
python InvData.5.diffusion.py
```

or, if you rename the file:

```bash
python main.py
```

The generated frames will be written to:

```text
./Video/
```

---

## Making a Movie

After the PNG files are generated, you can create a movie using `ffmpeg`:

```bash
ffmpeg -framerate 30 -i Video/%04d.png -c:v libx264 -pix_fmt yuv420p movie.mp4
```

---

## Notes on Numerical Method

- The solver uses Fourier transforms for spatial derivatives.
- The domain is periodic in both `x` and `y`.
- The flow is evolved in vorticity form.
- The tracer is stored and updated in Fourier space.
- Particle velocities are computed using periodic interpolation.
- Particle diffusion is modeled by Langevin noise.
- Particle injection is handled inside the small `dt` loop, not just once per output frame.

---


## Suggested Repository Layout

```text
.
├── InvData.5.diffusion.py
├── Wini.p
├── README.md
└── Video/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

---


<div align="center">

**Kolmogorov Flow · Passive Tracer · Stochastic Streaklines**

</div>
