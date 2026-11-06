"""D. Big-integer multiplication through an FFT convolution of the digits.

The Python counterpart of lib/fft_bench/problems/bignum_mul.rb: the same five
paths and the same gates, with Python's int in place of Ruby's Integer.
"""

import sys

import fixtures
import gates
from backend import to_host, xfft, xp, xsig
from problems import register

PATHS = ["fftconvolve", "rfft_pow2", "rfft_exact", "fft_pow2", "fft_exact"]


def coefficients(path, a, b):
    if path == "fftconvolve":
        return xsig.fftconvolve(a, b, mode="full")
    length = a.size + b.size - 1
    m = 1 << (length - 1).bit_length() if path.endswith("_pow2") else length
    if path.startswith("rfft"):
        return xfft.irfft(xfft.rfft(a, n=m) * xfft.rfft(b, n=m), n=m)[:length]
    return xfft.ifft(xfft.fft(a, n=m) * xfft.fft(b, n=m)).real[:length]


def width(base):
    w = len(str(base)) - 1
    if 10 ** w != base:
        raise ValueError(f"base {base} is not a power of 10")
    return w


def integer(host_digits, base):
    w = width(base)
    return int("".join(f"{int(d):0{w}d}" for d in host_digits[::-1]))


def digits(value, base, count):
    w = width(base)
    text = str(value).rjust(count * w, "0")
    if len(text) > count * w:
        raise ValueError(f"{value.bit_length()}-bit product has more than {count} digits")
    return [int(text[i:i + w]) for i in range(len(text) - w, -1, -w)]


def carry(coefficients, base, count):
    c = 0
    out = []
    for k in range(count):
        c, digit = divmod(c + (int(coefficients[k]) if k < len(coefficients) else 0), base)
        out.append(digit)
    return out, c


def mismatches(found, overflow, expected):
    return sum(x != y for x, y in zip(found, expected)) + (0 if overflow == 0 else 1)


def values(found):
    top = max((k for k, d in enumerate(found) if d != 0), default=-1)
    return {"product_digits": top + 1, "digit_sum": sum(found)}


class BignumMul:
    name = "bignum_mul"

    def fixture(self, size):
        return fixtures.read(self.name, size)

    def run(self, fixture):
        sys.set_int_max_str_digits(0)
        base = fixture.attr("base")
        product = integer(fixture["a"], base) * integer(fixture["b"], base)
        expected = digits(product, base, fixture["a"].size + fixture["b"].size)
        a = fixture.device("a")
        b = fixture.device("b")
        found = {path: coefficients(path, a, b).astype(xp.float64) for path in PATHS}
        carried = {path: carry(to_host(xp.rint(c)).tolist(), base, len(expected)) for path, c in found.items()}
        return {"exact": fixture.device("exact", xp.float64), "expected": expected, "coefficients": found,
                "carried": carried, "values": values(next(iter(carried.values()))[0])}

    def gate(self, result):
        found = result["coefficients"]
        out = []
        for path, c in found.items():
            digits_found, overflow = result["carried"][path]
            out.append(gates.lt(f"{path}.error", gates.max_abs(c - result["exact"]), 0.5))
            out.append(gates.eq(f"{path}.mismatches", mismatches(digits_found, overflow, result["expected"]), 0))
        cs = list(found.values())
        spread = max(gates.max_abs(x - y) for i, x in enumerate(cs) for y in cs[i + 1:])
        out.append(gates.lt("paths.spread", spread, 0.5))
        return out

    def bench_step(self, fixture, variant=None):
        path = variant or PATHS[0]
        if path not in PATHS:
            raise ValueError(f"bignum_mul has no variant {path}; known: {', '.join(PATHS)}")
        a = fixture.device("a")
        b = fixture.device("b")
        return lambda: xp.rint(coefficients(path, a, b))


register("bignum_mul", BignumMul())
