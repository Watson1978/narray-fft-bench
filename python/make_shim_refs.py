"""Builds data/shim_ref_small from data/shim_small with numpy and scipy.

The references test/test_numo_fft.rb checks FFTBench::NumoFFT against. Every
case of shim_small's case list becomes one array named after its id. numpy's
deprecation warnings are errors, so no case leans on a deprecated path.

  python/.venv/bin/python python/make_shim_refs.py
"""

import os
import sys
import warnings

import numpy
import numpy.fft
import scipy
import scipy.signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fixtures  # noqa: E402


def compute(case, inputs):
    args = [inputs[a] if isinstance(a, str) else a for a in case["args"]]
    if case["func"] == "fftconvolve":
        return scipy.signal.fftconvolve(*args, mode="full")
    return getattr(numpy.fft, case["func"])(*args, **case["kwargs"])


def main():
    shim = fixtures.read("shim", "small")
    refs = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for case in shim.attr("cases"):
            refs[case["id"]] = compute(case, shim.arrays)
    attrs = {"source": f"numpy {numpy.__version__} / scipy {scipy.__version__}",
             "shim_sha256": shim.manifest["sha256"]}
    manifest = fixtures.write("shim_ref", "small", refs, attrs)
    print(f"shim_ref_small: {manifest['bytes']} bytes, {len(refs)} cases, {attrs['source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
