"""Diagnose a common class-score shift using saved validation predictions only.

Margin = waterbird logit minus landbird logit. Align merge-on margins to merge-off
using one scalar median shift, estimated on other validation folds without labels
in the fitting objective. This is an explanatory control, not a new deployable method:
it needs full-token reference predictions for calibration. Checkpoint selection
already used this validation set, so the scores remain exploratory.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import GROUPS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auc(scores, labels):
    negatives = np.sort(scores[labels == 0])
    positives = scores[labels == 1]
    if not len(negatives) or not len(positives):
        raise ValueError("AUC requires both classes")
    lower = np.searchsorted(negatives, positives, side="left")
    upper = np.searchsorted(negatives, positives, side="right")
    return float(np.mean((lower + upper) / 2) / len(negatives))


def make_folds(groups, count=5, seed=317):
    rng = np.random.default_rng(seed)
    folds = np.full(len(groups), -1, dtype=int)
    if count < 2:
        raise ValueError("At least two calibration folds are required")
    for group in range(4):
        indices = np.flatnonzero(groups == group)
        if len(indices) < count:
            raise ValueError("Each group must have at least one example per fold")
        indices = rng.permutation(indices)
        folds[indices] = np.arange(len(indices)) % count
    if np.any(folds < 0):
        raise ValueError("Unexpected group labels")
    return folds


def align_margins(off, on, folds):
    aligned = np.empty_like(on)
    shifts = {}
    for fold in np.unique(folds):
        held_out = folds == fold
        shift = float(np.median((on-off)[~held_out]))
        aligned[held_out] = on[held_out] - shift
        shifts[str(fold)] = shift
    return aligned, shifts


def measures(scores, labels, groups):
    # Match torch.argmax's class-0 choice at an exactly zero margin.
    pred = scores > 0
    correct = pred == labels
    group_acc = {name: float(correct[groups == i].mean()) for i, name in enumerate(GROUPS)}
    return {"accuracy": float(correct.mean()), "wga": min(group_acc.values()),
            "group_accuracy": group_acc, "waterbird_prediction_fraction": float(pred.mean())}


def diagnose(off, on, labels, groups, folds):
    aligned, shifts = align_margins(off, on, folds)
    result = {"off": measures(off, labels, groups), "on": measures(on, labels, groups),
              "aligned_on_oof": measures(aligned, labels, groups), "fold_shifts": shifts,
              "auc_off": auc(off, labels), "auc_on": auc(on, labels), "delta_by_group": {},
              "auc_by_background": {}}
    delta = on - off
    for name, select in [("all", np.ones(len(labels), dtype=bool))] + [
            (name, groups == i) for i, name in enumerate(GROUPS)]:
        values = delta[select]
        result["delta_by_group"][name] = {"n": int(select.sum()), "median": float(np.median(values)),
            "mean": float(values.mean()), "q25": float(np.quantile(values, .25)),
            "q75": float(np.quantile(values, .75)), "fraction_negative": float(np.mean(values < 0))}
    for background, name in enumerate(("land", "water")):
        select = groups % 2 == background
        result["auc_by_background"][name] = {"off": auc(off[select], labels[select]),
                                              "on": auc(on[select], labels[select])}
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=317)
    p.add_argument("--out", default="runs/logit_shift_probe")
    args = p.parse_args(argv)
    rows, records, reference = [], [], None
    print("seed checkpoint   method   WGA_off WGA_on aligned_WGA AUC_off AUC_on median_delta")
    for seed in args.seeds:
        directory = Path(args.runs) / f"merge_probe_seed{seed}"
        meta = json.loads((directory / "results.json").read_text())
        if meta["split"] != "validation" or meta["seed"] != seed:
            raise ValueError("Expected the matching validation probe")
        for kind in ("fixed_epoch", "native_best"):
            for method in ("kd", "tome_kd"):
                path = directory / f"{kind}_{method}_validation_predictions.pt"
                predictions = torch.load(path, map_location="cpu", weights_only=True)
                a, b = predictions["merge_off"], predictions["merge_on"]
                for key in ("labels", "groups", "sample_ids"):
                    if not torch.equal(a[key], b[key]):
                        raise ValueError(f"Misaligned predictions: {path}")
                labels, groups, ids = (a[k].numpy() for k in ("labels", "groups", "sample_ids"))
                if reference is not None and any(not np.array_equal(x,y) for x,y in zip(reference, (labels,groups,ids))):
                    raise ValueError("Validation samples differ across checkpoints/seeds")
                reference = (labels, groups, ids)
                if len(np.unique(ids)) != len(ids) or not np.array_equal(labels, groups//2):
                    raise ValueError("Duplicate samples or invalid label/group mapping")
                margins = []
                for item in (a,b):
                    logits = item["logits"].double().numpy()
                    if logits.shape != (len(labels),2) or not np.isfinite(logits).all():
                        raise ValueError("Expected finite binary logits")
                    margins.append(logits[:,1]-logits[:,0])
                folds = make_folds(groups, args.folds, args.fold_seed)
                result = diagnose(*margins, labels, groups, folds)
                source = next(r for r in meta["records"] if r["method"] == method and r["checkpoint_kind"] == kind)
                row = {"seed": seed, "checkpoint_kind": kind, "method": method, "epoch": source["epoch"],
                       "wga_off_percent": 100*result["off"]["wga"], "wga_on_percent": 100*result["on"]["wga"],
                       "aligned_on_oof_wga_percent": 100*result["aligned_on_oof"]["wga"],
                       "auc_off": result["auc_off"], "auc_on": result["auc_on"],
                       "median_on_minus_off_margin": result["delta_by_group"]["all"]["median"]}
                rows.append(row)
                records.append({**row, "diagnostics": result, "predictions_sha256": sha(path),
                                "source_checkpoint_sha256": source["checkpoint_sha256"],
                                "sample_ids": ids.tolist(), "fold_ids": folds.tolist()})
                print(f"{seed:4d} {kind:12s} {method:7s} {row['wga_off_percent']:7.2f} {row['wga_on_percent']:6.2f} "
                      f"{row['aligned_on_oof_wga_percent']:11.2f} {row['auc_off']:.4f} {row['auc_on']:.4f} "
                      f"{row['median_on_minus_off_margin']:+.4f}", flush=True)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    with (output/"summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {"split": "validation", "probe_sha256": sha(__file__), "folds": args.folds,
              "fold_seed": args.fold_seed, "records": records,
              "interpretation": "Exploratory scalar-shift control using full-token reference predictions. "
              "Checkpoint selection already used validation. This is not independent test performance, "
              "proof of information preservation, or a causal gradient-loss test."}
    (output/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"Saved CPU-only diagnostic to {output}")


if __name__ == "__main__":
    main()
