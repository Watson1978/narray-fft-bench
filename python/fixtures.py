"""Reads the fixtures script/make_fixtures.rb writes, and writes the same format.

A complex array is two real blocks, entries with "part": "re" and "im".

  python/.venv/bin/python python/fixtures.py digest bignum_mul small
"""

import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("FFT_BENCH_DATA") or os.path.join(ROOT, "data")


class Fixture:
    def __init__(self, problem, size, arrays, attrs, manifest):
        self.problem = problem
        self.size = size
        self.arrays = arrays
        self.attrs = attrs
        self.manifest = manifest

    def __getitem__(self, name):
        return self.arrays[name]

    def attr(self, name):
        return self.attrs[name]

    def device(self, name, dtype=None):
        from backend import device
        return device(self.arrays[name], dtype)


def read(problem, size, data_dir=DATA_DIR):
    base = os.path.join(data_dir, f"{problem}_{size}")
    with open(base + ".json") as f:
        manifest = json.load(f)
    if manifest["dtype"] != "<f8" or manifest["order"] != "C":
        raise ValueError(f"{base}.json: unexpected layout {manifest['dtype']} {manifest['order']}")
    blocks = {}
    for entry in manifest["arrays"]:
        count = int(np.prod(entry["shape"], dtype=np.int64))
        flat = np.fromfile(base + ".bin", dtype="<f8", count=count, offset=entry["offset"])
        if flat.size != count:
            raise ValueError(f"{base}.bin is truncated at {entry['name']}")
        blocks.setdefault(entry["name"], {})[entry.get("part")] = flat.reshape(entry["shape"])
    arrays = {}
    for name, parts in blocks.items():
        if list(parts) == [None]:
            arrays[name] = parts[None]
        elif sorted(parts) == ["im", "re"]:
            z = np.empty(parts["re"].shape, dtype=np.complex128)
            z.real = parts["re"]
            z.imag = parts["im"]
            arrays[name] = z
        else:
            raise ValueError(f"{base}.json: {name} has parts {sorted(parts, key=str)}")
    return Fixture(problem, size, arrays, manifest["attrs"], manifest)


def write(problem, size, arrays, attrs=None, data_dir=DATA_DIR):
    base = os.path.join(data_dir, f"{problem}_{size}")
    entries = []
    offset = 0
    with open(base + ".bin", "wb") as f:
        for name, array in arrays.items():
            array = np.asarray(array)
            parts = [("re", array.real), ("im", array.imag)] if np.iscomplexobj(array) else [(None, array)]
            for part, values in parts:
                data = np.ascontiguousarray(values, dtype="<f8").tobytes()
                f.write(data)
                entry = {"name": name, "shape": list(array.shape), "offset": offset,
                         "sha256": hashlib.sha256(data).hexdigest()}
                if part:
                    entry["part"] = part
                entries.append(entry)
                offset += len(data)
    with open(base + ".bin", "rb") as f:
        digest_all = hashlib.sha256(f.read()).hexdigest()
    manifest = {"problem": problem, "size": size, "dtype": "<f8", "order": "C", "bytes": offset,
                "sha256": digest_all, "arrays": entries, "attrs": attrs or {}}
    with open(base + ".json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def _sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


def digest(problem, size):
    """sha256 of every block as read back, the complex ones split again."""
    fixture = read(problem, size)
    out = {}
    for name, array in fixture.arrays.items():
        if np.iscomplexobj(array):
            out[name] = {"shape": list(array.shape), "re": _sha(array.real), "im": _sha(array.imag)}
        else:
            out[name] = {"shape": list(array.shape), "sha256": _sha(array)}
    return out


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "digest":
        sys.exit("usage: fixtures.py digest <problem> <size>")
    json.dump(digest(sys.argv[2], sys.argv[3]), sys.stdout)
    print()
