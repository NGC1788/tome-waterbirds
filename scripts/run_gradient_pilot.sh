#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
SEED="${1:-0}"
DATA_DIR="${2:-data/waterbird_complete95_forest2water2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
OUT="runs/gradient_rescue_seed${SEED}"
mkdir -p "$OUT"
exec 9>"$OUT/queue.lock"
flock -n 9 || { echo "This gradient queue is already running."; exit 1; }
.venv/bin/python - "$SEED" <<'PY'
import json
from pathlib import Path
import sys
import run
seed = int(sys.argv[1])
cfg = json.loads(Path("config.json").read_text()); cfg["seed"] = seed
directory = Path("runs")/f"tome_kd_seed{seed}"
done = json.loads((directory/"training_complete.json").read_text())
spec = json.loads((directory/"manifest.json").read_text())["spec"]
if done["epochs"] != 100 or cfg != spec["config"] or spec["source_sha256"] != run.source_hash():
    raise ValueError("Use the completed baseline's unchanged 100-epoch training configuration/code")
if spec["teacher_sha256"] != run.sha("runs/teacher_seed0/best.pt"):
    raise ValueError("Teacher differs from the baseline")
if spec["runtime_versions"] != {"torch": str(run.torch.__version__), "timm": run.timm.__version__}:
    raise ValueError("torch/timm differs from the baseline")
print(f"Gradient pilot seed={seed}: baseline settings and fixed teacher verified.", flush=True)
PY

for arm in standard residual common shuffled; do
  resume_args=()
  if [[ -f "$OUT/$arm/manifest.json" ]]; then
    resume_args=(--resume)
  fi
  echo "[START] arm=$arm seed=$SEED strength=1"
  .venv/bin/python -u experiments/gradient_rescue.py train --data "$DATA_DIR" \
    --out "$OUT/$arm" --seed "$SEED" --arm "$arm" --strength 1 \
    "${resume_args[@]}" 2>&1 | tee -a "$OUT/$arm.log"
  echo "[FINISHED] arm=$arm seed=$SEED"
done
.venv/bin/python experiments/gradient_rescue.py report --out "$OUT"
echo "[ALL DONE] Gradient intervention pilot seed=$SEED completed; validation only."
