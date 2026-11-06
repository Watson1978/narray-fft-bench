#!/usr/bin/env bash
# Interleaves Ruby, Python and a control (Ruby again) within each round,
# rotating their order from round to round. 11 rounds (0-10); aggregate.py
# drops round 0. Every command prints the TSV line of script/bench.rb or
# python/bench.py, and the sixth column (ms/step) is what gets recorded.
#
# Each condition is name, mode (steady or first), Ruby command and Python
# command. The mode is appended to both commands as --mode, and the ninth
# column of their output has to echo it back.
#
#   bench/run.sh bench/conditions.tsv tmp/bench
#   python3 bench/aggregate.py bench/conditions.tsv tmp/bench
#
# When the conditions use the GPU, an unlocked SM clock stops the run before
# it starts (the lock does not survive a reboot). $out/monitor.log gets the
# load average and the busiest processes every 5 seconds, and on the GPU also
# the clocks and the number of GPU processes.
set -u
cd "$(dirname "$0")/.."

conditions=${1:?conditions file}
out=${2:?output directory}
rounds=${ROUNDS:-10}
export PYTHON=${PYTHON:-python/.venv/bin/python}
labels=(ruby python control)
mkdir -p "$out"

stop() {
  echo "STOPPED: $1 ($name, ${labels[$k]})"
  echo "  ${commands[$k]}"
  tail -5 "$out/last_stderr" | sed 's/^/  /'
  exit 1
}

gpu=0
if grep -v '^#' "$conditions" | grep -q 'GPU=1'; then
  gpu=1
  sm=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | head -1)
  if (( sm < 2700 )); then
    echo "STOPPED: idle SM clock is ${sm} MHz; lock it first (docs/method.md)"
    exit 1
  fi
fi
while :; do
  busiest=$(top -b -n 1 -w 200 -o %CPU | sed -n '8,10p' | awk '{printf "%s %s%%, ", $12, $9}')
  if (( gpu )); then
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" \
      "$(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader | head -1)" \
      "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) gpu procs" \
      "$(cut -d' ' -f1-3 /proc/loadavg)" "${busiest%, }"
  else
    printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$(cut -d' ' -f1-3 /proc/loadavg)" "${busiest%, }"
  fi
  sleep 5
done >> "$out/monitor.log" &
monitor=$!
trap 'kill $monitor 2>/dev/null' EXIT

while IFS=$'\t' read -r name mode ruby python; do
  [[ -z $name || $name == \#* ]] && continue
  if [[ $mode != steady && $mode != first ]]; then
    echo "STOPPED: $name: the mode column is '$mode', not steady or first"
    exit 1
  fi
  commands=("$ruby --mode $mode" "$python --mode $mode" "$ruby --mode $mode")
  file="$out/$name.tsv"
  : > "$file"
  for ((round = 0; round <= rounds; round++)); do
    for i in 0 1 2; do
      k=$(( (round + i) % 3 ))
      raw=$(timeout 1800 bash -c "${commands[$k]}" 2>"$out/last_stderr") || stop "exit $?"
      read -r value reported < <(printf '%s\n' "$raw" | awk -F'\t' 'NF >= 9 {v = $6; m = $9} END {print v, m}')
      [[ $value =~ ^[0-9]+(\.[0-9]+)?$ ]] || stop "no number in the output"
      [[ $reported == "$mode" ]] || stop "the output says mode '$reported', not $mode"
      printf '%s\t%s\t%s\n' "$round" "${labels[$k]}" "$value" >> "$file"
    done
  done
  echo "done $name"
done < "$conditions"
