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
