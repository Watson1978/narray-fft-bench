"""Runs the correctness gates of every problem and prints one JSON document.

The Python counterpart of script/gate.rb, with the same output shape.

  python/.venv/bin/python python/gate.py
  GPU=1 python/.venv/bin/python python/gate.py --only bignum_mul
"""

import argparse
import json
import os
import sys

os.environ.setdefault("DTYPE", "fp64")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import problems  # noqa: E402
from backend import backend_name, dtype_name  # noqa: E402
from gates import report  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="small")
    parser.add_argument("--only", type=lambda s: s.split(","))
    parser.add_argument("--out")
    args = parser.parse_args()

    registry = problems.load_all()
    names = args.only or list(registry)
    results = {}
    for name in names:
        problem = problems.fetch(name)
        try:
            result = problem.run(problem.fixture(args.size))
            gates = problem.gate(result)
            status = "pass" if all(gate.passed() for gate in gates) else "fail"
            results[name] = {"status": status, "gates": report(gates), "values": result.get("values", {})}
        except NotImplementedError as e:
            results[name] = {"status": "skipped", "reason": f"NotImplementedError: {e}"}

    doc = {"impl": "python", "backend": backend_name(), "dtype": dtype_name(),
           "size": args.size, "problems": results}
    text = json.dumps(doc, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    else:
        print(text)
    return 1 if any(r["status"] == "fail" for r in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
