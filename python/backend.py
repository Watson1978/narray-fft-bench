"""xp / xfft / xsig switch, the Python counterpart of lib/fft_bench/backend.rb."""

import os

GPU = os.environ.get("GPU", "").lower() in ("1", "on", "true")

if GPU:
    # FFTs do not use TF32, but the dots in D and F do.
    os.environ.setdefault("CUPY_TF32", "0")
    import cupy as xp
    import cupy.fft as xfft
    import cupyx.scipy.signal as xsig

    def synchronize():
        xp.cuda.Stream.null.synchronize()

    def plan_cache():
        return xfft.config.get_plan_cache()
else:
    import numpy as xp
    import numpy.fft as xfft
    import scipy.signal as xsig

    def synchronize():
        pass

    def plan_cache():
        return None

_DTYPES = {"": "float32", "fp32": "float32", "sfloat": "float32", "fp64": "float64", "dfloat": "float64"}
_dtype = os.environ.get("DTYPE", "").lower()
if _dtype not in _DTYPES:
    raise ValueError(f"unknown DTYPE {_dtype!r}; use fp32 or fp64")
XF = xp.dtype(_DTYPES[_dtype])
XC = xp.dtype("complex128" if XF == xp.float64 else "complex64")


def dtype_name():
    return "fp64" if XF == xp.float64 else "fp32"


def backend_name():
    return "cupy" if GPU else "numpy"


def device(host, dtype=None):
    if dtype is None:
        dtype = XC if xp.iscomplexobj(host) else XF
    return xp.asarray(host, dtype=dtype)


def to_host(array):
    return xp.asnumpy(array) if GPU else array


def scalar(value):
    return float(to_host(value))


def prime():
    """Loads cuFFT and builds a plan for a length no condition uses."""
    xfft.fft(xp.arange(3, dtype=xp.complex128))
    synchronize()


def describe():
    """One line that says what actually ran, printed by every bench."""
    if not GPU:
        import numpy
        import scipy
        return f"numpy {numpy.__version__} / scipy {scipy.__version__}"
    cache = plan_cache()
    return (f"cupy {xp.__version__} / CUPY_TF32={os.environ.get('CUPY_TF32')} / "
            f"plan cache {cache.get_curr_size()}/{cache.get_size()}")
