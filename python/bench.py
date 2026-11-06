"""Times one problem at one size in one dtype and prints a single TSV line.

The Python counterpart of script/bench.rb, with the same columns and modes:

  impl  backend  problem  size  dtype  ms/step  best_seconds  all_seconds  mode

  GPU=1 DTYPE=fp32 python/.venv/bin/python python/bench.py --problem bignum_mul --size n1e5 --inner 200
  GPU=1 DTYPE=fp32 python/.venv/bin/python python/bench.py --problem bignum_mul --size n1e5 --mode first
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import problems  # noqa: E402
from backend import backend_name, describe, dtype_name, prime, synchronize  # noqa: E402


def interval(step, inner):
    started = time.perf_counter()
    for _ in range(inner):
        step()
    synchronize()
    return time.perf_counter() - started


def measure(step, rounds, warmup, inner):
    for _ in range(warmup):
        for _ in range(inner):
            step()
    synchronize()
    return [interval(step, inner) for _ in range(rounds)]


def first(step):
    prime()
    return [interval(step, 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--variant")
    parser.add_argument("--mode", choices=["steady", "first"], default="steady")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--inner", type=int, default=1)
    args = parser.parse_args()

    problem = problems.fetch(args.problem)
    step = problem.bench_step(problem.fixture(args.size), args.variant)
    if args.mode == "first":
        times, inner = first(step), 1
    else:
        times, inner = measure(step, args.rounds, args.warmup, args.inner), args.inner
    best = min(times)
    name = f"{args.problem}:{args.variant}" if args.variant else args.problem
    print("\t".join([backend_name(), describe(), name, args.size, dtype_name(),
                     f"{best / inner * 1000:.4f}", f"{best:.6f}",
                     ",".join(f"{t:.6f}" for t in times), args.mode]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
