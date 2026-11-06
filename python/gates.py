"""Gate measures, the Python counterpart of lib/fft_bench/gates.rb.

Relative errors are max-norm: ‖x − x_ref‖_max / ‖x_ref‖_max.
"""

from backend import dtype_name, scalar, to_host, xp

DEFAULT_TOL = {"fp64": 1e-12, "fp32": 1e-5}


class Gate:
    def __init__(self, name, value, tol, op):
        self.name = name
        self.value = value
        self.tol = tol
        self.op = op

    def passed(self):
        if self.op == "le":
            return self.value <= self.tol
        if self.op == "lt":
            return self.value < self.tol
        if self.op == "ge":
            return self.value >= self.tol
        if self.op == "eq":
            return self.value == self.tol
        raise ValueError(f"unknown gate op {self.op!r}")

    def to_dict(self):
        return {"value": self.value, "tol": self.tol, "op": self.op, "pass": self.passed()}


def tol(dtype=None):
    return DEFAULT_TOL[dtype or dtype_name()]


def le(name, value, limit):
    return Gate(name, float(value), float(limit), "le")


def lt(name, value, limit):
    return Gate(name, float(value), float(limit), "lt")


def ge(name, value, limit):
    return Gate(name, float(value), float(limit), "ge")


def eq(name, value, expected):
    return Gate(name, value, expected, "eq")


def max_abs(a):
    return scalar(xp.abs(a).max())


def rel_max(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"shape differs: {actual.shape} vs {expected.shape}")
    return max_abs(actual - expected) / max_abs(expected)


def mismatches(actual, expected):
    a = to_host(actual)
    e = to_host(expected)
    if a.shape != e.shape:
        raise ValueError(f"shape differs: {a.shape} vs {e.shape}")
    return int((a != e).sum())


def sum_abs2(x):
    if xp.iscomplexobj(x):
        return scalar((x.real * x.real + x.imag * x.imag).sum())
    return scalar((x * x).sum())


def parseval(x, spectrum, n, norm=None, half=None):
    """Relative discrepancy of Σ|x|² = c Σ|X|², as in gates.rb."""
    energy = sum_abs2(x)
    c = {None: 1.0 / n, "backward": 1.0 / n, "ortho": 1.0, "forward": float(n)}[norm]
    return abs(energy - c * spectral_energy(spectrum, half)) / energy


def spectral_energy(spectrum, half):
    total = sum_abs2(spectrum)
    if half is None:
        return total
    edges = [0]
    if half % 2 == 0 and half > 1:
        edges.append(half // 2)
    return 2 * total - sum(sum_abs2(spectrum[..., k]) for k in edges)


def report(gates):
    return {gate.name: gate.to_dict() for gate in gates}
