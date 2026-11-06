"""Turns the per-round ms/step that bench/run.sh recorded into a table.

  python3 bench/aggregate.py bench/conditions.tsv tmp/bench [--labels Cumo,CuPy]

The ratio is Python ÷ Ruby per round, so above 1 means Ruby is faster. Each
ratio carries how many rounds exceeded 1 and a two-sided sign-test p. The
control column is control ÷ Ruby. It runs the same command as Ruby, so it
should cross 1; a condition whose control does not is biased by something
other than the implementation and stays out of the README.

The spread columns are max ÷ min of each series. Above 5%, read the raw
series in round order ($out/<name>.tsv) before trusting the median: a run
that splits into two groups puts the median between them.

Steady and first conditions go to separate tables, steady first: plan
creation is three orders of magnitude above a small steady transform, so the
two never share one.
"""

import argparse
import collections
import math
import os
import statistics

SPREAD_LIMIT = 0.05


def sign_test(above, n):
    tail = sum(math.comb(n, i) for i in range(min(above, n - above) + 1))
    return min(1.0, 2 * tail / 2**n)


def ratio(rows, rounds, numerator, denominator):
    values = [rows[r][numerator] / rows[r][denominator] for r in rounds]
    above = sum(v > 1 for v in values)
    text = f"{statistics.median(values):.3f} {above}/{len(values)} p={sign_test(above, len(values)):.2g}"
    if min(values) < 1 < max(values):
        text += " またぐ"
    return text


def spread(rows, rounds, label):
    values = [rows[r][label] for r in rounds]
    width = max(values) / min(values) - 1
    return f"{width * 100:.1f}%" + (" 要確認" if width > SPREAD_LIMIT else "")


def table(out, title, names, ruby_label, python_label):
    print(f"| {title} | {ruby_label} (ms) | {python_label} (ms) | {python_label} ÷ {ruby_label} | "
          f"対照 ÷ {ruby_label} | 幅 {ruby_label} | 幅 {python_label} |")
    print("|---|---|---|---|---|---|---|")
    for name in names:
        rows = collections.defaultdict(dict)
        for line in open(os.path.join(out, f"{name}.tsv")):
            round_, label, value = line.rstrip("\n").split("\t")
            if int(round_) >= 1:
                rows[int(round_)][label] = float(value)
        rounds = [r for r in sorted(rows) if len(rows[r]) == 3]
        median = {k: statistics.median(rows[r][k] for r in rounds) for k in ("ruby", "python")}
        print(f"| {name} | {median['ruby']:.3f} | {median['python']:.3f} | "
              f"{ratio(rows, rounds, 'python', 'ruby')} | {ratio(rows, rounds, 'control', 'ruby')} | "
              f"{spread(rows, rounds, 'ruby')} | {spread(rows, rounds, 'python')} |")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("conditions")
    parser.add_argument("out")
    parser.add_argument("--labels", default="Cumo,CuPy")
    args = parser.parse_args()
    ruby_label, python_label = args.labels.split(",")

    conditions = [line.split("\t", 2)[:2] for line in open(args.conditions)
                  if line.strip() and not line.startswith("#")]
    for mode, title in (("steady", "定常"), ("first", "初回")):
        names = [name for name, m in conditions if m == mode]
        if names:
            table(args.out, title, names, ruby_label, python_label)


if __name__ == "__main__":
    main()
