#!/usr/bin/env bash
# Linux server: repeat the frozen seed-0 protocol with student seeds 1 and 2.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
DATA_DIR="${1:-data/waterbird_complete95_forest2water2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p runs
exec 9>runs/repeat_students.lock
flock -n 9 || { echo "A repetition queue is already running."; exit 1; }

# Refuse accidental retuning or a different teacher between seed 0 and repeats.
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import run
cfg = json.loads(Path("config.json").read_text())
cfg["seed"] = 0
teacher_hash = run.sha("runs/teacher_seed0/best.pt")
for method in ("ce", "kd", "tome_kd"):
    directory = Path("runs") / f"{method}_seed0"
    completed = json.loads((directory / "training_complete.json").read_text())
    spec = json.loads((directory / "manifest.json").read_text())["spec"]
    if completed["epochs"] != 100 or cfg["student_epochs"] != 100:
        raise ValueError("Expected the completed 100-epoch seed-0 protocol")
    if cfg != spec["config"] or spec["source_sha256"] != run.source_hash():
        raise ValueError("Config or training source differs from seed 0")
    if spec["runtime_versions"] != {"torch": str(run.torch.__version__), "timm": run.timm.__version__}:
        raise ValueError("Use the same torch/timm versions as seed 0")
    if method != "ce" and spec["teacher_sha256"] != teacher_hash:
        raise ValueError("Teacher differs from seed 0")
print("Seed-0 protocol and fixed teacher verified.", flush=True)
PY

for seed in 1 2; do
  for method in ce kd tome_kd; do
    out="runs/${method}_seed${seed}"
    teacher_args=()
    resume_args=()
    if [[ "$method" != ce ]]; then
      teacher_args=(--teacher runs/teacher_seed0/best.pt)
    fi
    if [[ -f "$out/manifest.json" ]]; then
      resume_args=(--resume)
    fi
    echo "[START] $method seed=$seed"
    .venv/bin/python -u run.py train --data "$DATA_DIR" --out "$out" \
      --method "$method" --seed "$seed" "${teacher_args[@]}" "${resume_args[@]}" \
      2>&1 | tee -a "$out.log"
    echo "[FINISHED] $method seed=$seed"
  done
  .venv/bin/python -u scripts/validate_merge.py --data "$DATA_DIR" --runs runs \
    --seed "$seed" --out "runs/merge_probe_seed${seed}" \
    2>&1 | tee -a "runs/merge_probe_seed${seed}.log"
done
.venv/bin/python report.py --runs runs --out runs/report
echo "[ALL DONE] Student seeds 1 and 2, and validation merge probes completed."
