"""B. 2D Navier-Stokes in vorticity form, pseudo-spectral, Taylor-Green.

The Python counterpart of lib/fft_bench/problems/taylor_green.rb: the same
right-hand side, the same integrating-factor RK4 and the same gates.
"""

import math

import fixtures
import gates
from backend import XC, XF, dtype_name, scalar, xfft, xp
from problems import register

TAYLOR_TOL = {"fp64": 1e-8, "fp32": 5.2e-5}
CONSERVED_TOL = {"fp64": 1e-6, "fp32": 3.3e-6}
TOL = {"fp64": 4.9e-14, "fp32": 6e-6}


def abs2(z):
    return z.real * z.real + z.imag * z.imag


def grid(n, nu, dt):
    ky = xfft.fftfreq(n, d=1.0 / n).reshape(n, 1)
    kx = xfft.rfftfreq(n, d=1.0 / n).reshape(1, n // 2 + 1)
    k2 = (ky * ky + kx * kx).astype(XF)
    safe = k2.copy()
    safe[0, 0] = 1
    inverse = 1 / safe
    inverse[0, 0] = 0

    def keep(ks):
        return xp.asarray([1 if abs(k) < n / 3.0 else 0 for k in ks], dtype=XF)

    ky_values = [i if i < n // 2 else i - n for i in range(n)]
    mask = keep(ky_values).reshape(n, 1) * keep(range(n // 2 + 1)).reshape(1, n // 2 + 1)
    return {"n": n, "dt": dt, "ikx": kx.astype(XC) * 1j, "iky": ky.astype(XC) * 1j, "inverse": inverse,
            "mask": mask, "full": xp.exp(k2 * (-nu * dt)), "half": xp.exp(k2 * (-nu * dt / 2))}


def rhs(w, g):
    shape = (g["n"], g["n"])
    psi = w * g["inverse"]
    u = xfft.irfft2(psi * g["iky"], s=shape)
    v = xfft.irfft2(-(psi * g["ikx"]), s=shape)
    omega = xfft.irfft2(w, s=shape)
    flux = g["ikx"] * xfft.rfft2(u * omega) + g["iky"] * xfft.rfft2(v * omega)
    return -(flux * g["mask"])


def step(w, g):
    dt = g["dt"]
    a = rhs(w, g) * dt
    b = rhs(g["half"] * (w + a * 0.5), g) * dt
    c = rhs(g["half"] * w + b * 0.5, g) * dt
    d = rhs(g["full"] * w + g["half"] * c, g) * dt
    return g["full"] * w + (g["full"] * a + 2 * g["half"] * (b + c) + d) / 6


def spectral_sum(z, n):
    return 2 * abs2(z).sum() - abs2(z[:, 0]).sum() - abs2(z[:, n // 2]).sum()


def evolve(omega, g, steps):
    n = g["n"]
    w = xfft.rfft2(omega)
    energies, enstrophies, parseval = [], [], []
    for i in range(steps + 1):
        if i > 0:
            w = step(w, g)
        real = xfft.irfft2(w, s=(n, n))
        z = spectral_sum(w, n)
        energies.append(spectral_sum(w * xp.sqrt(g["inverse"]), n))
        enstrophies.append(z)
        parseval.append(((real * real).sum() * (n * n), z))
    return {"energy": [scalar(e) for e in energies], "enstrophy": [scalar(e) for e in enstrophies],
            "parseval": [abs(scalar(r) - scalar(z)) / scalar(z) for r, z in parseval]}


def drift(series):
    return abs(series[-1] - series[0]) / series[0]


def checks(omega, n):
    spectrum = xfft.rfft2(omega)
    shifted = xfft.fftshift(xfft.fftfreq(n, d=1.0 / n))
    return {"round_trip": gates.rel_max(xfft.irfft2(spectrum, s=(n, n)), omega),
            "transposed": gates.rel_max(xfft.rfft2(omega.T, axes=(-1, -2)).T, spectrum),
            "fftshift": gates.mismatches(shifted, xp.arange(n, dtype=xp.float64) - n // 2)}


def values(taylor, random):
    return {"taylor.peak": taylor["peak"], "random.energy": random["energy"][0],
            "random.energy_end": random["energy"][-1], "random.enstrophy": random["enstrophy"][0],
            "random.enstrophy_end": random["enstrophy"][-1]}


class TaylorGreen:
    name = "taylor_green"

    def fixture(self, size):
        return fixtures.read(self.name, size)

    def run(self, fixture):
        spec = fixture.attr("spec")
        omega = fixture.device("omega_random")
        n = omega.shape[0]
        g = grid(n, spec["nu"], spec["dt"])
        taylor_field = fixture.device("omega_taylor")
        w = xfft.rfft2(taylor_field)
        for _ in range(spec["steps"]):
            w = step(w, g)
        exact = taylor_field * math.exp(-2 * spec["nu"] * spec["dt"] * spec["steps"])
        final = xfft.irfft2(w, s=(n, n))
        taylor = {"error": gates.rel_max(final, exact), "peak": scalar(final.max())}
        random = evolve(omega, grid(n, 0.0, spec["dt"]), spec["steps"])
        return {"taylor": taylor, "random": random, "checks": checks(omega, n), "values": values(taylor, random)}

    def gate(self, result):
        dtype = dtype_name()
        tol = TOL[dtype]
        r = result["random"]
        out = [gates.le("taylor.error", result["taylor"]["error"], TAYLOR_TOL[dtype]),
               gates.le("random.energy_drift", drift(r["energy"]), CONSERVED_TOL[dtype]),
               gates.le("random.enstrophy_drift", drift(r["enstrophy"]), CONSERVED_TOL[dtype]),
               gates.le("random.parseval", max(r["parseval"]), tol)]
        for name, value in result["checks"].items():
            out.append(gates.eq(name, value, 0) if name == "fftshift" else gates.le(name, value, tol))
        return out

    def bench_step(self, fixture, variant=None):
        if variant not in (None, "rk4"):
            raise ValueError(f"taylor_green has no variant {variant}")
        spec = fixture.attr("spec")
        omega = fixture.device("omega_random")
        g = grid(omega.shape[0], spec["nu"], spec["dt"])
        w = xfft.rfft2(omega)
        return lambda: step(w, g)


register("taylor_green", TaylorGreen())
