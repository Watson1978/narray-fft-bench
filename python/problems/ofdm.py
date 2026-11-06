"""F. OFDM over a multipath channel.

The Python counterpart of lib/fft_bench/problems/ofdm.rb: the same link, the
same subcarrier-averaged error rate and the same gates.
"""

import math

import fixtures
import gates
from backend import XC, XF, dtype_name, to_host, xfft, xp, xsig
from problems import register

NORMS = ["backward", "ortho", "forward"]
VARIANTS = ["batch", "single"]
SIGMAS = 3.0
PARSEVAL_TOL = {"fp64": 5.6e-14, "fp32": 6.1e-7}


def abs2(z):
    return z.real * z.real + z.imag * z.imag


def qpsk(bits):
    count, width = bits.shape
    pairs = bits.reshape(count, width // 2, 2)
    re = 1 - 2 * pairs[:, :, 0]
    im = 1 - 2 * pairs[:, :, 1]
    return (re.astype(XC) + im.astype(XC) * 1j) / math.sqrt(2)


def decide(z):
    count, n = z.shape
    out = xp.zeros((count, n, 2), dtype=XF)
    out[:, :, 0] = (1 - z.real / xp.abs(z.real)) * 0.5
    out[:, :, 1] = (1 - z.imag / xp.abs(z.imag)) * 0.5
    return out.reshape(count, 2 * n)


def prefixed(x, cp):
    n = x.shape[-1]
    out = xp.zeros((*x.shape[:-1], n + cp), dtype=x.dtype)
    out[..., :cp] = x[..., n - cp:]
    out[..., cp:] = x
    return out


def channel(x, h):
    return xsig.fftconvolve(x, h.reshape(1, h.size), mode="full")[:, :x.shape[1]]


def demodulate(y, cp, norm, response):
    return xfft.fft(y[..., cp:], norm=norm) / response


def noisy_errors(fixture, symbols, bits, h, response, gains, cp):
    if "noise" not in fixture.arrays:
        return {}
    noise = fixture.device("noise")
    received = channel(prefixed(xfft.ifft(symbols, norm="ortho"), cp), h)
    per_subcarrier = 2 * symbols.shape[0]
    out = {}
    for db in fixture.attr("spec")["ebn0_db"]:
        ebn0 = 10 ** (db / 10.0)
        equalized = demodulate(received + noise * math.sqrt(1 / (2 * ebn0)), cp, "ortho", response)
        errors = gates.mismatches(decide(equalized), bits)
        p = [0.5 * math.erfc(math.sqrt(g * ebn0)) for g in gains]
        expected = per_subcarrier * sum(p)
        spread = math.sqrt(per_subcarrier * sum(q * (1 - q) for q in p))
        out[db] = {"errors": errors, "expected": expected, "sigmas": abs(errors - expected) / spread}
    return out


def shift_checks(equalized, n):
    frequencies = xfft.fftshift(xfft.fftfreq(n))
    ascending = (xp.arange(n, dtype=xp.float64) - n // 2) / n
    constellation = xfft.fftshift(equalized, axes=[-1])
    return {"frequencies": gates.mismatches(frequencies, ascending),
            "dc_center": gates.mismatches(constellation[:, n // 2], equalized[:, 0])}


def values(clean, noisy, gains):
    out = {"clean_errors": sum(c["errors"] for c in clean.values()), "channel_gain": sum(gains) / len(gains)}
    for db, x in noisy.items():
        out[f"awgn.{db}dB.errors"] = x["errors"]
    return out


class Ofdm:
    name = "ofdm"

    def fixture(self, size):
        return fixtures.read(self.name, size)

    def run(self, fixture):
        cp = fixture.attr("cp")
        bits = fixture.device("bits")
        symbols = qpsk(bits)
        n = symbols.shape[1]
        h = fixture.device("channel")
        response = xfft.fft(h, n=n)
        clean = {}
        for norm in NORMS:
            tx = xfft.ifft(symbols, norm=norm)
            equalized = demodulate(channel(prefixed(tx, cp), h), cp, norm, response)
            clean[norm] = {"errors": gates.mismatches(decide(equalized), bits),
                           "parseval": gates.parseval(tx, symbols, n, norm=norm), "equalized": equalized}
        gains = to_host(abs2(response)).tolist()
        noisy = noisy_errors(fixture, symbols, bits, h, response, gains, cp)
        return {"clean": clean, "noisy": noisy, "shift": shift_checks(clean["ortho"]["equalized"], n),
                "values": values(clean, noisy, gains)}

    def gate(self, result):
        out = []
        for norm, c in result["clean"].items():
            out.append(gates.eq(f"{norm}.bit_errors", c["errors"], 0))
            out.append(gates.le(f"{norm}.parseval", c["parseval"], PARSEVAL_TOL[dtype_name()]))
        out.extend(gates.le(f"awgn.{db}dB.sigmas", x["sigmas"], SIGMAS) for db, x in result["noisy"].items())
        out.extend(gates.eq(f"fftshift.{name}", value, 0) for name, value in result["shift"].items())
        return out

    def bench_step(self, fixture, variant=None):
        name = variant or VARIANTS[0]
        if name not in VARIANTS:
            raise ValueError(f"ofdm has no variant {name}; known: {', '.join(VARIANTS)}")
        cp = fixture.attr("cp")
        symbols = qpsk(fixture.device("bits"))
        response = xfft.fft(fixture.device("channel"), n=symbols.shape[1])

        def link(x):
            return demodulate(prefixed(xfft.ifft(x, norm="ortho"), cp), cp, "ortho", response)

        if name == "batch":
            return lambda: link(symbols)
        rows = [symbols[i] for i in range(symbols.shape[0])]
        return lambda: [link(x) for x in rows]


register("ofdm", Ofdm())
