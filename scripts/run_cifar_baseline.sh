#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
seed="${1:-0}"
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "Seed must be a nonnegative integer"; exit 1; }
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
out="runs/cifar100_baseline"
mkdir -p "$out"
exec 9>"$out/.queue_seed${seed}.lock"
flock -n 9 || { echo "This seed queue is already running"; exit 1; }
py=.venv/bin/python
"$py" experiments/cifar_baseline.py prepare
"$py" experiments/cifar_baseline.py preflight
for method in ce kd; do
  echo "[START] ${method} seed=${seed}; detailed log: $out/${method}_seed${seed}.log"
  "$py" experiments/cifar_baseline.py train \
    --method "$method" --seed "$seed" --out "$out/${method}_seed${seed}" --resume \
    2>&1 | tee -a "$out/${method}_seed${seed}.log"
  echo "[DONE] ${method} seed=${seed}"
done
"$py" experiments/cifar_baseline.py finish --seed "$seed" --runs "$out"
echo "[ALL DONE] Results: $out/report_seed${seed}/summary.csv"
