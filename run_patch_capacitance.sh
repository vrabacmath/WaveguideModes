#!/usr/bin/env bash
set -euo pipefail

program="./bin/WaveguideModes"

if [[ ! -x "$program" ]]; then
  echo "Error: $program does not exist or is not executable." >&2
  exit 1
fi

for n in {2..6}; do
  for e in 2; do
    echo "$program bent-waveguide $n $n 1 64 $e"
    "$program" bent-waveguide "$n" "$n" 1 64 "$e"
  done
done

for c in 0 1 2; do
  for e in 2; do
    echo "$program bent-waveguide 7 7 $c 64 $e"
    "$program" bent-waveguide 7 7 "$c" 64 "$e"
  done
done

#!/usr/bin/env bash
set -euo pipefail

program="./bin/WaveguideModes"

if [[ ! -x "$program" ]]; then
  echo "Error: $program does not exist or is not executable." >&2
  exit 1
fi

e=2
echo "$program patch-capacitance 2 2 64 1 $e"
"$program" patch-capacitance 2 2 64 "$e"

echo "$program patch-capacitance 3 2 64 1 $e"
"$program" patch-capacitance 3 2 64 "$e"

echo "$program patch-capacitance 5 3 64 1 $e"
"$program" patch-capacitance 5 2 64 "$e"

echo "$program patch-capacitance 6 4 64 1 $e"
"$program" patch-capacitance 6 2 64 "$e"

echo "$program patch-capacitance 6 4 64 1 $e"
"$program" patch-capacitance 6 3 64 "$e"

echo "$program patch-capacitance 6 4 64 1 $e"
"$program" patch-capacitance 6 4 64 "$e"

echo "$program patch-capacitance 10 4 64 1 $e"
"$program" patch-capacitance 10 4 64 "$e"