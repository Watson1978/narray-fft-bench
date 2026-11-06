"""G. Correlation and spectral estimates (Wiener-Khinchin, hfft, Welch).

The Python counterpart of lib/fft_bench/problems/spectrum.rb: the same paths,
the same Welch estimate written by hand (not scipy.signal.welch) and the same
gates.
"""

import math

import fixtures
import gates
from backend import XC, XF, dtype_name, scalar, xfft, xp, xsig
from problems import register

KINDS = ["pow2", "mixed", "prime"]
SEGMENTS = [256, 1024, 4096]
CYCLE = [(length, kind) for length in [128, 256, 512, 1024, 2048, 4096] for kind in KINDS]
VARIANTS = [v for kind in KINDS for v in [f"wk_{kind}", *[f"welch{length}_{kind}" for length in SEGMENTS]]]
VARIANTS.append("welch_cycle")
GATE_SEGMENT = 256
TOL = {"fp64": 1.5e-14, "fp32": 6e-6}
STATISTICAL_TOL = 0.1
GUARD = 4


def abs2(z):
    return z.real * z.real + z.imag * z.imag


def autocorrelation(x):
    n = x.size
    return xfft.irfft(abs2(xfft.rfft(x, n=2 * n)), n=2 * n)[:n]


def transforms(x):
    n = x.size
    r = autocorrelation(x)
    circular = r.copy()
    circular[1:] = r[1:] + r[1:][::-1]
    half = circular[:n // 2 + 1]
    spectrum = xfft.hfft(half, n=n)
    return {"r": r, "convolved": xsig.fftconvolve(x, x[::-1], mode="full")[n - 1:2 * n - 1],
            "complex": xfft.ifft(abs2(xfft.fft(x, n=2 * n))).real[:n],
            "hfft": spectrum[:n // 2 + 1], "power": abs2(xfft.rfft(x)), "ihfft": xfft.ihfft(spectrum),
            "half": half.astype(XC)}


def hann(length):
    return 0.5 - 0.5 * xp.cos(xp.arange(length, dtype=XF) * (2 * math.pi / length))


def one_sided(length):
    w = xp.full(length // 2 + 1, 2, dtype=XF)
    w[0] = 1
    w[-1] = 1
    return w


# The segments that start at multiples of L split into two runs, one from 0
# and one from L / 2, each a plain reshape.
def welch(x, length, fs):
    window = hann(length)
    runs = []
    for start in (0, length // 2):
        count = (x.size - start) // length
        runs.append(x[start:start + count * length].reshape(count, length))
    power = sum(abs2(xfft.rfft(segments * window, axis=-1)).sum(axis=0) for segments in runs)
    count = sum(segments.shape[0] for segments in runs)
    return power * one_sided(length) / (fs * (3.0 * length / 8) * count)


def welch_stats(x, spec):
    psd = welch(x, GATE_SEGMENT, spec["fs"])
    peak = int(scalar(xp.argmax(psd)))
    freqs = xfft.rfftfreq(GATE_SEGMENT, d=1.0 / spec["fs"])
    rest = [psd[1:peak - GUARD], psd[peak + GUARD + 1:-1]]
    return {"peak_frequency": scalar(freqs[peak]), "peak_height": scalar(psd[peak]),
            "noise_floor": sum(scalar(part.sum()) for part in rest) / sum(part.size for part in rest)}


def floor(spec):
    return 2 * spec["sigma"] ** 2 / spec["fs"]


def expected_peak(spec, length):
    return spec["amplitude"] ** 2 / 2 * (length / 2.0) ** 2 / (spec["fs"] * 3.0 * length / 8) + floor(spec)


def values(all_transforms):
    out = {}
    for kind, t in all_transforms.items():
        out[f"{kind}.energy"] = scalar(t["long"]["r"][0])
        for name, value in t["welch"].items():
            out[f"{kind}.{name}"] = value
    return out


class Spectrum:
    name = "spectrum"

    def fixture(self, size):
        return fixtures.read(self.name, size)

    def run(self, fixture):
        spec = fixture.attr("spec")
        all_transforms = {}
        for kind in KINDS:
            short = transforms(fixture.device(f"short_{kind}"))
            short["direct"] = fixture.device(f"short_{kind}_direct")
            long = fixture.device(kind)
            all_transforms[kind] = {"short": short, "long": transforms(long), "welch": welch_stats(long, spec)}
        return {"transforms": all_transforms, "spec": spec, "values": values(all_transforms)}

    def gate(self, result):
        spec = result["spec"]
        tol = TOL[dtype_name()]
        out = []
        for kind, t in result["transforms"].items():
            short = t["short"]
            out.append(gates.le(f"{kind}.short.direct", gates.rel_max(short["r"], short["direct"]), tol))
            for length in ("short", "long"):
                x = t[length]
                out.append(gates.le(f"{kind}.{length}.fftconvolve", gates.rel_max(x["convolved"], x["r"]), tol))
                out.append(gates.le(f"{kind}.{length}.complex", gates.rel_max(x["complex"], x["r"]), tol))
                out.append(gates.le(f"{kind}.{length}.hfft", gates.rel_max(x["hfft"], x["power"]), tol))
                out.append(gates.le(f"{kind}.{length}.ihfft", gates.rel_max(x["ihfft"], x["half"]), tol))
            w = t["welch"]
            out.append(gates.eq(f"{kind}.welch.peak_frequency", w["peak_frequency"], spec["f0"]))
            out.append(gates.le(f"{kind}.welch.peak_height",
                                abs(w["peak_height"] / expected_peak(spec, GATE_SEGMENT) - 1), STATISTICAL_TOL))
            out.append(gates.le(f"{kind}.welch.noise_floor", abs(w["noise_floor"] / floor(spec) - 1), STATISTICAL_TOL))
        return out

    def bench_step(self, fixture, variant=None):
        name = variant or VARIANTS[0]
        if name not in VARIANTS:
            raise ValueError(f"spectrum has no variant {name}; known: {', '.join(VARIANTS)}")
        fs = fixture.attr("spec")["fs"]
        signals = {kind: fixture.device(kind) for kind in KINDS}
        if name == "welch_cycle":
            return lambda: [welch(signals[kind], length, fs) for length, kind in CYCLE]
        op, kind = name.split("_")
        x = signals[kind]
        if op == "wk":
            return lambda: autocorrelation(x)
        length = int(op.removeprefix("welch"))
        return lambda: welch(x, length, fs)


register("spectrum", Spectrum())
