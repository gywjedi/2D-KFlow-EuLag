#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:07:00 2026

@author: yawei
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:12:44 2026

@author: yawei
"""

import os
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator


class Evolve:
    def __init__(self, Nx, Ny, n, Re, dt, D=0.001, a=0.1, sigma=0.05):
        self.Nx = Nx
        self.Ny = Ny
        self.n = n
        self.Re = Re
        self.dt = dt
        self.D = D
        self.a = a
        self.sigma = sigma

        self.Lx = 2 * np.pi
        self.Ly = 2 * np.pi

        self.x = np.linspace(0, self.Lx - self.Lx / Nx, Nx)
        self.y = np.linspace(0, self.Ly - self.Ly / Ny, Ny)

        self.kx, self.ky = np.meshgrid(
            np.fft.fftfreq(Nx, d=1 / Nx),
            np.fft.fftfreq(Ny, d=1 / Ny),
            indexing='ij'
        )

        # For implementing the forcing
        self.kron_kx = np.zeros((Nx, Ny))
        self.kron_kx[0, :] = 1

        self.kron_ky = np.zeros((Nx, Ny))
        self.kron_ky[:, n] = 1
        self.kron_ky[:, -n] = 1

        self.force = (self.n / 2) * self.kron_kx * self.kron_ky

        # Vorticity-to-velocity conversion matrices
        alphainner = self.ky[1:, 1:] / self.kx[1:, 1:]
        onemat = np.ones((self.Nx - 1, self.Ny - 1))

        d1 = 1j * self.ky[1:, 1:] * alphainner
        d2 = 1j * self.kx[1:, 1:]

        self.d3 = onemat / (d2 + d1)
        self.d4 = -alphainner * self.d3

        # Navier-Stokes linear operator
        self.G = (self.kx**2 + self.ky**2) / Re
        self.inv = 1 / (1 / self.dt + self.G / 2)
        self.lin = 1 / self.dt - self.G / 2

        # Tracer diffusion linear operator
        self.Gc = self.D * (self.kx**2 + self.ky**2)
        self.invc = 1 / (1 / self.dt + self.Gc / 2)
        self.linc = 1 / self.dt - self.Gc / 2

        # Work arrays
        self.Fu = np.zeros((Nx, Ny), complex)
        self.Fv = np.zeros((Nx, Ny), complex)

    def fft2(self, q):
        return (1 / self.Nx) * (1 / self.Ny) * np.fft.fftn(q, axes=(0, 1))

    def ifft2(self, q_hat):
        return self.Nx * self.Ny * np.real(np.fft.ifftn(q_hat, axes=(0, 1)))

    def init_C(self):
        C = np.zeros((self.Nx, self.Ny))
        C_hat = self.fft2(C)
        return C, C_hat

    def w2u(self, w):
        W = self.fft2(w)
        return self.W2u(W)

    def W2u(self, W):
        # Reset to avoid old modes accidentally remaining
        self.Fu[:, :] = 0
        self.Fv[:, :] = 0

        # Case kx = 0, ky != 0
        self.Fu[0, 1:] = (-1 / (1j * self.ky[0, 1:])) * W[0, 1:]

        # Case kx != 0, ky = 0
        self.Fv[1:, 0] = (1 / (1j * self.kx[1:, 0])) * W[1:, 0]

        # Case kx != 0, ky != 0
        self.Fu[1:, 1:] = self.d4 * W[1:, 1:]
        self.Fv[1:, 1:] = self.d3 * W[1:, 1:]

        u = self.ifft2(self.Fu)
        v = self.ifft2(self.Fv)

        return u, v

    def u2w(self, u, v):
        u_fft = self.fft2(u)
        v_fft = self.fft2(v)

        W = 1j * self.kx * v_fft - 1j * self.ky * u_fft
        w = self.ifft2(W)

        return w

    def forcing2f(self):
        x, y = np.meshgrid(self.x, self.y, indexing='ij')

        forcing = self.a * np.exp(
            -(((x - np.pi) ** 2) + ((y - np.pi) ** 2)) / (2 * self.sigma**2)
        )

        forcing_hat = self.fft2(forcing)
        return forcing_hat

    def C2f(self, C_hat, W, forcing_hat):
        C = self.ifft2(C_hat)
        u, v = self.W2u(W)

        fftuC = self.fft2(u * C)
        fftvC = self.fft2(v * C)

        adv_hat = -1j * ((self.kx * fftuC) + (self.ky * fftvC))

        return adv_hat + forcing_hat

    def W2f(self, W):
        w = self.ifft2(W)
        u, v = self.W2u(W)

        fftuw = self.fft2(u * w)
        fftvw = self.fft2(v * w)

        f = -1j * (self.kx * fftuw + self.ky * fftvw) - self.force

        return f

    def dWdt(self, W, T):
        tsteps = round(T / self.dt)

        for _ in range(tsteps):
            Wf = self.W2f(W)

            invlinW = self.inv * self.lin * W

            W1 = invlinW + self.inv * Wf
            W = invlinW + 0.5 * self.inv * (self.W2f(W1) + Wf)

        return W

    def AdvectParticles(self, xs, W):
        """
        Advect all particles by one solver timestep self.dt using the
        current velocity field from W.

        The interpolation grid is extended periodically to x = 2*pi and
        y = 2*pi. This avoids extrapolating near the periodic boundary.
        """
        u, v = self.W2u(W)

        # Extend the velocity field periodically so interpolation is valid
        # on [0, 2*pi] instead of only [0, 2*pi - dx].
        x_ext = np.r_[self.x, self.Lx]
        y_ext = np.r_[self.y, self.Ly]

        u_ext = np.empty((self.Nx + 1, self.Ny + 1))
        v_ext = np.empty((self.Nx + 1, self.Ny + 1))

        u_ext[:-1, :-1] = u
        v_ext[:-1, :-1] = v

        u_ext[-1, :-1] = u[0, :]
        v_ext[-1, :-1] = v[0, :]

        u_ext[:-1, -1] = u[:, 0]
        v_ext[:-1, -1] = v[:, 0]

        u_ext[-1, -1] = u[0, 0]
        v_ext[-1, -1] = v[0, 0]

        ui = RegularGridInterpolator(
            (x_ext, y_ext),
            u_ext,
            bounds_error=False,
            fill_value=None
        )

        vi = RegularGridInterpolator(
            (x_ext, y_ext),
            v_ext,
            bounds_error=False,
            fill_value=None
        )

        for i in range(len(xs)):
            xq = xs[i][0] % self.Lx
            yq = xs[i][1] % self.Ly

            ux = ui([[xq, yq]]).squeeze()
            vx = vi([[xq, yq]]).squeeze()

            xs[i][0] = (xq + ux * self.dt) % self.Lx
            xs[i][1] = (yq + vx * self.dt) % self.Ly

        return xs
    
    def AdvectionDiffusionParticles(self, xs, W, D_particles=0.0, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        u, v = self.W2u(W)

        # Extend the velocity field periodically so interpolation is valid
        # on [0, 2*pi] instead of only [0, 2*pi - dx].
        x_ext = np.r_[self.x, self.Lx]
        y_ext = np.r_[self.y, self.Ly]

        u_ext = np.empty((self.Nx + 1, self.Ny + 1))
        v_ext = np.empty((self.Nx + 1, self.Ny + 1))

        u_ext[:-1, :-1] = u
        v_ext[:-1, :-1] = v

        u_ext[-1, :-1] = u[0, :]
        v_ext[-1, :-1] = v[0, :]

        u_ext[:-1, -1] = u[:, 0]
        v_ext[:-1, -1] = v[:, 0]

        u_ext[-1, -1] = u[0, 0]
        v_ext[-1, -1] = v[0, 0]

        ui = RegularGridInterpolator(
            (x_ext, y_ext),
            u_ext,
            bounds_error=False,
            fill_value=None
        )

        vi = RegularGridInterpolator(
            (x_ext, y_ext),
            v_ext,
            bounds_error=False,
            fill_value=None
        )

        noise_amp = np.sqrt(2.0 * D_particles * self.dt)

        for i in range(len(xs)):
            xq = xs[i][0] % self.Lx
            yq = xs[i][1] % self.Ly

            ux = ui([[xq, yq]]).squeeze()
            vx = vi([[xq, yq]]).squeeze()

            dx_diff = noise_amp * rng.normal()
            dy_diff = noise_amp * rng.normal()

            xs[i][0] = (xq + ux * self.dt + dx_diff) % self.Lx
            xs[i][1] = (yq + vx * self.dt + dy_diff) % self.Ly

        return xs

    def dCdt_CN_Heunn_with_particles(self, C_hat, W, forcing_hat, xs, T, D_particles = 0, seed=None, inject_every=10):
        """
        Coupled update.

        Every small dt:
            1. update W
            2. update C_hat
            3. update xs using the updated W
        """
        tsteps = round(T / self.dt)
        rng = np.random.default_rng(seed)

        for _ in range(tsteps):
            # --------------------------
            # Update vorticity W
            # --------------------------
            Wf = self.W2f(W)
            invlinW = self.inv * self.lin * W

            W1 = invlinW + self.inv * Wf
            W_next = invlinW + 0.5 * self.inv * (self.W2f(W1) + Wf)

            # --------------------------
            # Update tracer C_hat
            # --------------------------
            fc = self.C2f(C_hat, W, forcing_hat)
            C_next = self.invc * (self.linc * C_hat + fc)

            fcf = self.C2f(C_next, W_next, forcing_hat)
            C_hat = self.invc * (self.linc * C_hat + 0.5 * (fc + fcf))

            # --------------------------
            # Update particles xs
            # --------------------------
            #xs = self.AdvectParticles(xs, W_next)  only diffusion
            xs = self.AdvectionDiffusionParticles(xs, W_next, D_particles=D_particles, rng=rng)

            W = W_next

        return C_hat, W, xs

    def Plotting(self, u, title=''):
        plt.pcolormesh(
            self.x,
            self.y,
            u.T,
            vmin=-5,
            vmax=5,
            shading='gouraud'
        )
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------
    Nx = 128
    Ny = 128
    dt = 0.001

    Re = 14.4
    n = 2
    D = 1e-3

    n_frames = 2000
    dT = 0.25

    # -------------------------------------------------------------------------
    # Initial condition
    # -------------------------------------------------------------------------
    x = np.linspace(0, 2 * np.pi - 2 * np.pi / Nx, Nx)
    y = np.linspace(0, 2 * np.pi - 2 * np.pi / Ny, Ny)

    u0 = (Re / n**2) * np.sin(n * y)
    u0 = np.repeat(u0[np.newaxis, :], Nx, axis=0)

    v0 = np.zeros((Nx, Ny))

    KFlow = Evolve(Nx, Ny, n, Re, dt, D=D)

    np.random.seed(0)
    w0 = KFlow.u2w(u0, v0) + 0.01 * Re * np.random.randn(Nx, Ny)

    W = KFlow.fft2(w0)

    # -------------------------------------------------------------------------
    # Load the same spun-up field used by InvData.3.py
    # -------------------------------------------------------------------------
    # Important: do NOT regenerate or overwrite Wini.p here.
    W = pickle.load(open('Wini.p', 'rb'))

    # -------------------------------------------------------------------------
    # Initialize tracer and particles
    # -------------------------------------------------------------------------
    Cs = []
    Ws = []
    times = []
    particle_history = []

    C, C_hat = KFlow.init_C()
    forcing_hat = KFlow.forcing2f()

    xs = [[np.pi, np.pi]]

    # -------------------------------------------------------------------------
    # Main evolution loop
    # -------------------------------------------------------------------------
    for frame in range(n_frames):

        print(frame)

        # Save current field and particles at the same time
        Cs.append(KFlow.ifft2(C_hat))
        Ws.append(KFlow.ifft2(W))
        times.append(frame * dT)
        particle_history.append(copy.deepcopy(xs))

        # Inject new particle every 10 output frames
        if frame % 10 == 0:
            xs.append([np.pi, np.pi])

        # Coupled update:
        # C_hat, W, and xs are all updated together inside the small-dt loop.
        C_hat, W, xs = KFlow.dCdt_CN_Heunn_with_particles(
            C_hat,
            W,
            forcing_hat,
            xs,
            dT
        )

    # -------------------------------------------------------------------------
    # Plot tracer concentrations and streakline
    # -------------------------------------------------------------------------
    os.makedirs("./Video", exist_ok=True)

    sp = 4

    for i in range(1, len(Cs) - 1):
        plt.figure()

        Cplot = Cs[i]
        Cmax = np.max(Cplot)

        if Cmax != 0:
            Cplot = Cplot / Cmax

        plt.pcolormesh(
            KFlow.x,
            KFlow.y,
            Cplot.T,
            cmap='Blues',
            shading='gouraud',
            vmin=0,
            vmax=1
        )

        plt.colorbar()

        pts = np.asarray(particle_history[i])

        if len(pts) > 0:
            plt.scatter(
                pts[:, 0],
                pts[:, 1],
                marker="o",
                s=3,
                color='r',
                label="streakline"
            )

        u, v = KFlow.w2u(Ws[i])

        plt.quiver(
            KFlow.x[::sp],
            KFlow.y[::sp],
            u[::sp, ::sp].T,
            v[::sp, ::sp].T,
            color='k',
            scale=50
        )

        plt.title(f'Tracer concentration at t = {times[i]:.2f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(0, 2 * np.pi)
        plt.ylim(0, 2 * np.pi)

        plt.savefig(f'./Video/{i:04d}.png', dpi=150)
        plt.close()