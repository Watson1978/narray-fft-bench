"""One module per problem (A-G), one-to-one with lib/fft_bench/problems/.

Each module calls register() with an object that has fixture(size), run(fixture),
gate(result) and bench_step(fixture, variant), the same four as the Ruby side.
"""

import importlib
import pkgutil

REGISTRY = {}


def register(name, problem):
    REGISTRY[name] = problem
    return problem


def load_all():
    for info in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{info.name}")
    return REGISTRY


def fetch(name):
    load_all()
    if name not in REGISTRY:
        raise KeyError(f"unknown problem {name!r}; known: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[name]
