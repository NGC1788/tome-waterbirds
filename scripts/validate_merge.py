"""Validation-only merge on/off probe of existing, completed student runs.

Compare a common training epoch first; native best-WGA checkpoints are secondary.
No training, test-set scoring, checkpoint selection, or checkpoint mutation.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run
import torch
from torch.nn import functional as F
from data import GROUPS, Waterbirds, audit
from models import build


def paired_stats(off, on):
    for key in ("sample_ids", "labels", "groups"):
        if not torch.equal(off[key], on[key]):
            raise ValueError(f"Unpaired evaluation: {key}")
    a, b = off["logits"].argmax(-1), on["logits"].argmax(-1)
    ca, cb = a.eq(off["labels"]), b.eq(off["labels"])
    kl = F.kl_div(F.log_softmax(on["logits"], -1), F.log_softmax(off["logits"], -1),
                  log_target=True, reduction="none").sum(-1)
    result = {}
    for group, select in [("all", torch.ones_like(ca))] + [
            (name, off["groups"].eq(i)) for i, name in enumerate(GROUPS)]:
        if not select.any():
            raise ValueError(f"Missing group: {group}")
        result[group] = {"n": int(select.sum()), "prediction_flips": int((a.ne(b) & select).sum()),
                         "correct_off_wrong_on": int((ca & ~cb & select).sum()),
                         "wrong_off_correct_on": int((~ca & cb & select).sum()),
                         "kl_off_to_on_temperature1": float(kl[select].mean())}
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--runs", default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--out", default="runs/merge_probe_seed0")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    info = audit(args.data)
    validation = Waterbirds(args.data, 1)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    rows, records, reference = [], [], None
    for kind in ("fixed_epoch", "native_best"):
        for method in ("kd", "tome_kd"):
            directory = Path(args.runs) / f"{method}_seed{args.seed}"
            complete = json.loads((directory / "training_complete.json").read_text())
            if complete["epochs"] < args.epoch:
                raise ValueError(f"{directory}: requested epoch has not completed")
            path = directory / (f"epoch_{args.epoch:03d}.pt" if kind == "fixed_epoch" else "best.pt")
            state = torch.load(path, map_location="cpu", weights_only=True)
            spec = state["spec"]
            if spec["role"] != "student" or spec["method"] != method or spec["config"]["seed"] != args.seed:
                raise ValueError(f"Checkpoint role/method/seed mismatch: {path}")
            if kind == "fixed_epoch" and state["epoch"] != args.epoch:
                raise ValueError(f"Checkpoint epoch mismatch: {path}")
            for key in ("metadata_sha256", "images_sha256"):
                if spec[key] != info[key]:
                    raise ValueError(f"Checkpoint data mismatch: {path}")
            if spec["source_sha256"] != run.source_hash():
                raise ValueError("Training source changed; restore the original training files")
            shared = {k: spec[k] for k in ("config", "teacher_sha256", "runtime_versions")}
            if reference is not None and shared != reference:
                raise ValueError("KD and ToMe-KD config/teacher/runtime mismatch")
            reference = shared
            if spec["runtime_versions"] != {"torch": str(torch.__version__), "timm": run.timm.__version__}:
                raise ValueError("Use the original training torch/timm versions")
            cfg = spec["config"]
            run.seed_all(cfg["seed"])
            model = build("student", cfg, pretrained=False).to(device)
            model.load_state_dict(state["model"])
            values, predictions = {}, {}
            for merge in (False, True):
                mode = "merge_on" if merge else "merge_off"
                m, pred = run.evaluate(model, validation, cfg, device, state["train_weights"],
                                       merge=merge, return_predictions=True)
                values[mode], predictions[mode] = m, pred
                row = {"checkpoint_kind": kind, "method": method, "epoch": state["epoch"],
                       "mode": mode, "accuracy_percent": 100*m["accuracy"], "wga_percent": 100*m["wga"]}
                row.update({name+"_percent": 100*m["group_accuracy"][name] for name in GROUPS})
                rows.append(row)
                print(f"{kind:12s} {method:7s} epoch={state['epoch']:3d} {mode:9s} "
                      f"accuracy={m['accuracy']:.2%} WGA={m['wga']:.2%}", flush=True)
            record = {"checkpoint_kind": kind, "method": method, "epoch": state["epoch"],
                      "checkpoint": str(path), "checkpoint_sha256": run.sha(path), "spec": spec,
                      "metrics": values, "paired": paired_stats(predictions["merge_off"], predictions["merge_on"])}
            records.append(record)
            run.save(output / f"{kind}_{method}_validation_predictions.pt", predictions)
            del model, state
    with (output / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    run.write_json(output / "results.json", {"split": "validation", "seed": args.seed, "data": info,
                   "environment": run.environment(device), "probe_sha256": run.sha(__file__),
                   "records": records, "interpretation": "Diagnostic only; merge-off is a distribution change "
                   "for a model trained with merging. This does not isolate gradient loss causally."})
    print(f"Saved validation-only probe to {output}")


if __name__ == "__main__":
    main()
