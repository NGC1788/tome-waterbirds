"""Export observed validation histories; never invent or extrapolate results."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",default="runs")
    p.add_argument("--out",default="runs/report")
    args=p.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for history_path in sorted(Path(args.runs).rglob("history.json")):
        directory=history_path.parent
        manifest=json.loads((directory/"manifest.json").read_text())
        history=json.loads(history_path.read_text())
        if not history: continue
        spec=manifest["spec"]
        # Earliest maximum, identical to training selection.
        best=max(history,key=lambda r:r["validation"]["wga"])
        row={"run":str(directory),"method":spec["method"],"seed":spec["config"]["seed"],
             "epochs_finished":history[-1]["epoch"],"best_epoch":best["epoch"],
             "best_val_wga_percent":100*best["validation"]["wga"],
             "best_val_accuracy_percent":100*best["validation"]["accuracy"],
             "training_seconds":sum(r["train"]["seconds"] for r in history),
             "complete":(directory/"training_complete.json").exists()}
        for group,value in best["validation"]["group_accuracy"].items():
            row["val_"+group+"_percent"]=100*value
        rows.append(row)
        fig,axes=plt.subplots(1,2,figsize=(10,3.5))
        epochs=[r["epoch"] for r in history]
        axes[0].plot(epochs,[100*r["validation"]["wga"] for r in history],label="Validation WGA")
        axes[0].plot(epochs,[100*r["validation"]["accuracy"] for r in history],label="Validation accuracy")
        train=[r for r in history if "train_eval" in r]
        axes[0].plot([r["epoch"] for r in train],[100*r["train_eval"]["accuracy"] for r in train],label="Train (eval transform)")
        for group in best["validation"]["group_accuracy"]:
            axes[1].plot(epochs,[100*r["validation"]["group_accuracy"][group] for r in history],label=group)
        for ax in axes:
            ax.set(xlabel="Epoch",ylabel="Accuracy (%)",ylim=(0,100)); ax.legend(fontsize=7); ax.grid(alpha=.2)
        fig.suptitle(f"{spec['method']} | seed {spec['config']['seed']} | validation only")
        fig.tight_layout()
        import hashlib
        suffix=hashlib.sha256(str(directory).encode()).hexdigest()[:8]
        fig.savefig(out/f"{spec['method']}_seed{spec['config']['seed']}_{suffix}.png",dpi=160)
        plt.close(fig)
    if not rows:
        raise SystemExit("No observed training histories found. No report generated.")
    with (out/"validation_summary.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"Exported {len(rows)} observed runs to {out}")


if __name__ == "__main__":
    main()
