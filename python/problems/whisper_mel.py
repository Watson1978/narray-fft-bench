"""A. Whisper's log-mel spectrogram.

The Python counterpart of lib/fft_bench/problems/whisper_mel.rb: the same
framing, the same batched rfft against the same DFT tables, the same clamps.
"""

import math

import fixtures
import gates
from backend import XF, dtype_name, scalar, xfft, xp
from problems import register

VARIANTS = ["rfft", "dft"]
FLOOR = 1.0e-10
DYNAMIC_RANGE = 8.0
PATHS_TOL = {"fp64": 1e-10, "fp32": 1.1e-2}
PARSEVAL_TOL = {"fp64": 1.4e-11, "fp32": 4.5e-7}


def abs2(z):
    return z.real * z.real + z.imag * z.imag


def prepare(fixture):
    n_fft = fixture.attr("n_fft")
    bins = n_fft // 2 + 1
    n = xp.arange(n_fft, dtype=XF).reshape(1, n_fft)
    angle = xp.arange(n_fft, dtype=XF).reshape(n_fft, 1) * xp.arange(bins, dtype=XF).reshape(1, bins) * XF.type(
        -2.0 * math.pi / n_fft)
    return {"n_fft": n_fft, "hop": fixture.attr("hop_length"), "frames": fixture.attr("nb_max_frames"), "bins": bins,
            "rate": fixture.attr("sampling_rate"), "filters": fixture.device("mel_filters"),
            "window": 0.5 - 0.5 * xp.cos(n * XF.type(2.0 * math.pi / n_fft)), "cos": xp.cos(angle), "sin": xp.sin(angle)}


def reflect(signal, pad):
    size = signal.size
    out = xp.zeros(size + 2 * pad, dtype=XF)
    out[pad:pad + size] = signal
    out[:pad] = signal[1:pad + 1][::-1]
    out[pad + size:] = signal[size - pad - 1:size - 1][::-1]
    return out


def frame(waveform, s):
    padded = reflect(waveform, s["n_fft"] // 2)
    blocks, rest = divmod(s["n_fft"], s["hop"])
    needed = s["frames"] + blocks + (0 if rest == 0 else 1) - 1
    grid = padded[:needed * s["hop"]].reshape(needed, s["hop"])
    out = xp.zeros((s["frames"], s["n_fft"]), dtype=XF)
    for b in range(blocks):
        out[:, b * s["hop"]:(b + 1) * s["hop"]] = grid[b:b + s["frames"], :]
    if rest:
        out[:, blocks * s["hop"]:] = grid[blocks:blocks + s["frames"], :rest]
    return out * s["window"]


def dft_power(frames, s):
    real = frames.dot(s["cos"])
    imag = frames.dot(s["sin"])
    return real * real + imag * imag


def log_mel(power, s):
    spec = xp.log10(xp.clip(s["filters"].dot(power.T), FLOOR, None))
    floor = spec.max() - DYNAMIC_RANGE
    return ((spec + floor + xp.abs(spec - floor)) * 0.5 + 4.0) / 4.0


class WhisperMel:
    name = "whisper_mel"

    def fixture(self, size):
        return fixtures.read(self.name, size)

    def run(self, fixture):
        s = prepare(fixture)
        frames = frame(fixture.device("waveform"), s)
        spectrum = xfft.rfft(frames, n=s["n_fft"], axis=-1)
        rfft_mel = log_mel(abs2(spectrum), s)
        dft_mel = log_mel(dft_power(frames, s), s)
        freqs = xfft.rfftfreq(s["n_fft"], d=1.0 / s["rate"])
        spacing = xp.arange(s["bins"], dtype=xp.float64) * (s["rate"] / s["n_fft"])
        return {"paths": gates.rel_max(rfft_mel, dft_mel),
                "parseval": gates.parseval(frames, spectrum, s["n_fft"], half=s["n_fft"]),
                "frequencies": gates.rel_max(freqs, spacing),
                "values": {"frames": rfft_mel.shape[1], "mean": scalar(rfft_mel.mean()), "max": scalar(rfft_mel.max()),
                           "min": scalar(rfft_mel.min())}}

    def gate(self, result):
        dtype = dtype_name()
        return [gates.le("paths", result["paths"], PATHS_TOL[dtype]),
                gates.le("parseval", result["parseval"], PARSEVAL_TOL[dtype]), gates.le("frequencies", result["frequencies"], 0)]

    def bench_step(self, fixture, variant=None):
        path = variant or VARIANTS[0]
        if path not in VARIANTS:
            raise ValueError(f"whisper_mel has no variant {path}; known: {', '.join(VARIANTS)}")
        s = prepare(fixture)
        waveform = fixture.device("waveform")
        if path == "rfft":
            return lambda: log_mel(abs2(xfft.rfft(frame(waveform, s), n=s["n_fft"], axis=-1)), s)
        return lambda: log_mel(dft_power(frame(waveform, s), s), s)


register("whisper_mel", WhisperMel())
