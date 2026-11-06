"""C. The potential of a Gaussian charge in a periodic box.

The Python counterpart of lib/fft_bench/problems/poisson3d.rb: the same three
paths, the same Ewald reference in the fixture and the same gates.
"""

import math

import fixtures
import gates
from backend import XF, dtype_name, scalar, xfft, xp
from problems import register

VARIANTS = ["rfftn", "fftn", "axes01"]
REFERENCE_TOL = 1e-6
LAPLACIAN_TOL = {"fp64": 4e-13, "fp32": 2.9e-4}
TOL = {"fp64": 2.8e-15, "fp32": 3.2e-6}


def inverse(k2):
    safe = k2.copy()
    safe[(0,) * k2.ndim] = 1
    out = 1 / safe
    out[(0,) * k2.ndim] = 0
    return out


def grid(n, box):
    k = (xfft.fftfreq(n, d=box / n) * (2 * math.pi)).astype(XF)
    half = (xfft.rfftfreq(n, d=box / n) * (2 * math.pi)).astype(XF)
    k0 = k.reshape(n, 1, 1)
    k1 = k.reshape(1, n, 1)
    both = k0 * k0 + k1 * k1
    half2 = both + (half * half).reshape(1, 1, half.size)
    full2 = both + (k * k).reshape(1, 1, n)
    return {"inverse_half": inverse(half2), "inverse_full": inverse(full2), "minus_k2": -half2, "n": n,
            "inverse_slice": inverse(k0 * k0 + (half * half).reshape(1, half.size, 1))}


def solve(rho, g, path):
    if path == "rfftn":
        return xfft.irfftn(xfft.rfftn(rho) * g["inverse_half"], s=rho.shape)
    if path == "fftn":
        return xfft.ifftn(xfft.fftn(rho) * g["inverse_full"]).real
    return xfft.irfftn(xfft.rfftn(rho, axes=(0, 1)) * g["inverse_slice"], s=rho.shape[:2], axes=(0, 1))


def potentials(phi, points):
    c = phi.shape[0] // 2
    return [scalar(phi[c + int(p[0]), c + int(p[1]), c + int(p[2])]) for p in points]


def difference_error(found, reference):
    df = [v - found[0] for v in found]
    dr = [v - reference[0] for v in reference]
    return max(abs(a - b) for a, b in zip(df, dr)) / max(abs(b) for b in dr)


class Poisson3d:
    name = "poisson3d"

    def fixture(self, size):
        return fixtures.read(self.name, size)

    def run(self, fixture):
        spec = fixture.attr("spec")
        rho = fixture.device("rho")
        n = rho.shape[0]
        g = grid(n, spec["box"])
        phi = solve(rho, g, "rfftn")
        found = potentials(phi, fixture["points"].tolist())
        reference = fixture["reference"].tolist()
        odd = fixture.device("rho_odd")
        return {"reference": difference_error(found, reference),
                "complex": gates.rel_max(solve(rho, g, "fftn"), phi),
                "laplacian": gates.rel_max(xfft.irfftn(xfft.rfftn(phi) * g["minus_k2"], s=rho.shape), -(rho - rho.mean())),
                "round_trip": gates.rel_max(xfft.irfftn(xfft.rfftn(rho), s=rho.shape), rho),
                "odd_round_trip": gates.rel_max(xfft.irfftn(xfft.rfftn(odd), s=odd.shape), odd),
                "axes01": gates.rel_max(xfft.fft(xfft.rfftn(rho, axes=(0, 1)), axis=2), xfft.rfftn(rho, axes=(2, 0, 1))),
                "values": {"phi_span": max(found) - min(found), "phi_axis_8": found[2] - found[0],
                           "odd_length_without_s": xfft.irfftn(xfft.rfftn(odd)).shape[-1]}}

    def gate(self, result):
        dtype = dtype_name()
        return [gates.le("reference", result["reference"], REFERENCE_TOL),
                gates.le("laplacian", result["laplacian"], LAPLACIAN_TOL[dtype])] + [
            gates.le(name, result[name], TOL[dtype]) for name in ["complex", "round_trip", "odd_round_trip", "axes01"]]

    def bench_step(self, fixture, variant=None):
        path = variant or VARIANTS[0]
        if path not in VARIANTS:
            raise ValueError(f"poisson3d has no variant {path}; known: {', '.join(VARIANTS)}")
        rho = fixture.device("rho")
        g = grid(rho.shape[0], fixture.attr("spec")["box"])
        return lambda: solve(rho, g, path)


register("poisson3d", Poisson3d())
