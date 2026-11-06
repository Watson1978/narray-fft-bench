"""E. Image registration by phase correlation.

The Python counterpart of lib/fft_bench/problems/phase_correlation.rb: the same
two paths, the same three-point sinc estimate and the same gates.
"""

import fixtures
import gates
from backend import dtype_name, scalar, to_host, xfft, xp
from problems import register

VARIANTS = ["fft2", "rfft2"]
DIFFER_TOL = {"fp64": 2.2e-15, "fp32": 2.4e-6}
PEAK_TOL = {"fp64": 1e-10, "fp32": 1.2e-6}
FRACTION_TOL = 0.05


def surface(i, j, path):
    if path == "rfft2":
        cross = xfft.rfft2(i) * xp.conj(xfft.rfft2(j))
        return xfft.fftshift(xfft.irfft2(cross / xp.abs(cross), s=i.shape))
    cross = xfft.fft2(i) * xp.conj(xfft.fft2(j))
    return xfft.fftshift(xfft.ifft2(cross / xp.abs(cross)).real)


def peak_index(r):
    return divmod(int(to_host(xp.argmax(r))), r.shape[1])


def locate(i, j, truth):
    surfaces = {path: surface(i, j, path) for path in VARIANTS}
    found = {}
    for path, r in surfaces.items():
        p = peak_index(r)
        found[path] = {"shift": [n // 2 - pk for pk, n in zip(p, r.shape)], "peak": scalar(r.max())}
    found["truth"] = truth
    found["differ"] = gates.rel_max(surfaces["rfft2"], surfaces["fft2"])
    return found


def offset(left, top, right):
    return right / (right + top) if right > left else -left / (left + top)


# The peak is never on the border: every shift is under n / 4.
def estimate(r, truth):
    p0, p1 = peak_index(r)
    column = to_host(r[p0 - 1:p0 + 2, p1]).tolist()
    row = to_host(r[p0, p1 - 1:p1 + 2]).tolist()
    found = [n // 2 - p - offset(*three) for p, three, n in zip((p0, p1), (column, row), r.shape)]
    return {"shift": found, "error": max(abs(f - t) for f, t in zip(found, truth))}


def values(integer, fractional):
    out = {"shifts_found": sum(f["fft2"]["shift"] == f["truth"] for f in integer),
           "peak_min": min(f["fft2"]["peak"] for f in integer)}
    for k, f in enumerate(fractional):
        for axis, s in enumerate(f["shift"]):
            out[f"fractional.{k}.axis{axis}"] = s
    return out


class PhaseCorrelation:
    name = "phase_correlation"

    def fixture(self, size):
        return fixtures.read(self.name, size)

    def run(self, fixture):
        images = {"even": fixture.device("image")}
        if "image_odd" in fixture.arrays:
            images["odd"] = fixture.device("image_odd")
        shifts = [[int(s) for s in row] for row in fixture["shifts"].tolist()]
        integer = {name: [locate(image, xp.roll(image, s, axis=(0, 1)), s) for s in shifts]
                   for name, image in images.items()}
        image = images["even"]
        frac_images = fixture.device("frac_images")
        fractional = [estimate(surface(image, frac_images[k], "fft2"), truth)
                      for k, truth in enumerate(fixture["frac_shifts"].tolist())]
        round_trips = {name: gates.mismatches(xfft.fftshift(xfft.ifftshift(x)), x)
                       + gates.mismatches(xfft.ifftshift(xfft.fftshift(x)), x) for name, x in images.items()}
        return {"integer": integer, "fractional": fractional, "round_trips": round_trips,
                "values": values(integer["even"], fractional)}

    def gate(self, result):
        out = []
        for name, found in result["integer"].items():
            wrong = sum(f[v]["shift"] != f["truth"] for f in found for v in VARIANTS)
            out.append(gates.eq(f"{name}.wrong_shifts", wrong, 0))
            out.append(gates.le(f"{name}.paths_differ", max(f["differ"] for f in found), DIFFER_TOL[dtype_name()]))
            out.append(gates.ge(f"{name}.peak_min", min(f["fft2"]["peak"] for f in found), 1 - PEAK_TOL[dtype_name()]))
        out.append(gates.le("fractional.error", max(f["error"] for f in result["fractional"]), FRACTION_TOL))
        out.extend(gates.eq(f"{name}.shift_round_trip", count, 0) for name, count in result["round_trips"].items())
        return out

    def bench_step(self, fixture, variant=None):
        path = variant or VARIANTS[0]
        if path not in VARIANTS:
            raise ValueError(f"phase_correlation has no variant {path}; known: {', '.join(VARIANTS)}")
        image = fixture.device("image")
        shifted = xp.roll(image, [int(s) for s in fixture["shifts"][0]], axis=(0, 1))
        return lambda: xp.argmax(surface(image, shifted, path))


register("phase_correlation", PhaseCorrelation())
