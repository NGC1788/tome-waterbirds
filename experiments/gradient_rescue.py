"""Cost-matched backward intervention pilot; main student forward is always ToMe.

At the single merge site all input token masses are one. P = unmerge(mean(grad))
is the orthogonal projection onto group-constant gradients. A detached full-token
suffix supplies g_full; residual = (I-P) g_full is the component ordinary mean
merging cannot propagate. This is an expensive diagnostic, not an efficient method.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as base
import torch
from data import loader

ARMS = ("standard", "residual", "common", "shuffled")


def full_suffix(model, x):
    """Resume immediately after the merge site's attention residual, without merging."""
    net = model.net
    block = net.blocks[model.merge_block]
    x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
    for block in net.blocks[model.merge_block+1:]:
        x = block(x)
    return net.forward_head(net.norm(x))


def full_reference_gradient(model, before, labels, targets, cfg, device):
    # Isolate dropout RNG; autograd.grad requests ONLY the detached boundary input.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        with torch.autocast(device_type=device.type, enabled=False):
            x = before.detach().float().requires_grad_(True)
            logits = full_suffix(model, x)
            loss, _, _ = base.loss_fn(logits, labels, targets.detach(), cfg)
            grad, = torch.autograd.grad(loss, x)
    if not torch.isfinite(grad).all():
        raise FloatingPointError("Nonfinite full-reference boundary gradient")
    return grad.detach(), loss.detach()


def shuffled_within_groups(value, group_ids, num_groups, seed):
    """Nonzero random cyclic shifts: preserve each group's sum and vector multiset."""
    b, n, c = value.shape
    counts = torch.zeros(b, num_groups, device=value.device, dtype=torch.long)
    counts.scatter_add_(1, group_ids, torch.ones_like(group_ids))
    starts = counts.cumsum(1) - counts
    order = group_ids.argsort(dim=1, stable=True)
    sorted_groups = group_ids.gather(1, order)
    generator = torch.Generator(device=value.device).manual_seed(seed)
    rand = torch.rand(b, num_groups, device=value.device, generator=generator)
    offsets = (rand * (counts-1).clamp(min=1)).long() + 1
    offsets = torch.where(counts > 1, offsets, 0)
    start = starts.gather(1, sorted_groups)
    rank = torch.arange(n, device=value.device)[None, :] - start
    shifted_rank = (rank + offsets.gather(1, sorted_groups)) % counts.gather(1, sorted_groups)
    source = order.gather(1, start + shifted_rank)
    shuffled = value.gather(1, source[..., None].expand(b,n,c))
    return torch.zeros_like(value).scatter(1, order[..., None].expand(b,n,c), shuffled)


@torch.no_grad()
def intervention_vectors(grad, trace, seed):
    merge, unmerge, size = trace["merge"], trace["unmerge"], trace["size"].float()
    projected = unmerge(merge(grad, mode="sum") / size)
    residual = grad - projected
    participating = unmerge(size) > 1
    common = projected * participating
    residual_norm = residual.flatten(1).norm(dim=1)
    common_norm = common.flatten(1).norm(dim=1)
    scale = torch.where(common_norm > 0, residual_norm/common_norm.clamp_min(1e-30), 0.)
    common = common * scale[:,None,None]
    ids = unmerge(torch.arange(size.shape[1], device=grad.device)[None,:,None].expand(size.shape[0],-1,-1)).squeeze(-1)
    shuffled = shuffled_within_groups(residual, ids, size.shape[1], seed)
    energy = (grad * participating).square().flatten(1).sum(1)
    fraction = residual.square().flatten(1).sum(1) / energy.clamp_min(1e-30)
    diagnostics = {"residual_energy_fraction": fraction.mean(),
                   "residual_norm": residual_norm.mean(),
                   "common_zero_norm_fraction": ((common_norm == 0) & (residual_norm > 0)).float().mean()}
    return {"residual": residual, "common": common, "shuffled": shuffled}, diagnostics


def inject_gradient(loss, before, vector, strength):
    # Exactly zero added forward value; only the boundary-input derivative changes.
    surrogate = (before.float() * vector.detach()).sum() * strength
    return loss + (surrogate - surrogate.detach())


def train_epoch(model, teacher, dataset, cfg, device, optimizer, scaler, epoch, merge):
    if teacher is None or not merge or model.merge_r <= 0:
        raise ValueError("The gradient intervention requires KD and a single nonempty merge")
    settings = cfg["gradient_intervention"]
    arm, strength = settings["arm"], settings["strength"]
    base.seed_all(cfg["seed"] + epoch*100003)
    dataset.epoch = epoch
    batches = loader(dataset, cfg, train=True)
    model.train(); teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    outputs, labels, groups = [], [], []
    sums = dict(loss=0., ce=0., kd=0., reference_loss=0., residual_energy_fraction=0.,
                residual_norm=0., common_zero_norm_fraction=0.)
    skipped, seen, updates = 0, 0, 0
    base.synchronize(device)
    started = time.monotonic()
    for step, (x, y, g, _) in enumerate(batches):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with base.autocast(cfg, device):
            with torch.no_grad():
                targets = teacher(x, merge=False)
            logits, trace = model(x, merge=True, capture=True)
            loss, ce, kd = base.loss_fn(logits, y, targets, cfg)
        reference_grad, reference_loss = full_reference_gradient(model, trace["before"], y, targets, cfg, device)
        vectors, diagnostics = intervention_vectors(reference_grad, trace,
                                      cfg["seed"] + epoch*100003 + step*10000019)
        # Every arm computes the same reference and candidate vectors. Only this changes.
        objective = loss if arm == "standard" else inject_gradient(loss, trace["before"], vectors[arm], strength)
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Nonfinite objective: {epoch=} {step=}")
        first = step // cfg["accumulation_steps"] * cfg["accumulation_steps"]
        last = min(first+cfg["accumulation_steps"], len(batches))
        samples = min(last*cfg["batch_size"], len(dataset)) - first*cfg["batch_size"]
        scaler.scale(objective * len(x)/samples).backward()
        if step+1 == last:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=not scaler.is_enabled())
            scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            skip = scaler.get_scale() < scale
            skipped += int(skip); updates += int(not skip)
            optimizer.zero_grad(set_to_none=True)
        values = {"loss": loss, "ce": ce, "kd": kd, "reference_loss": reference_loss, **diagnostics}
        for key, value in values.items():
            sums[key] += float(value.detach()) * len(x)
        outputs.append(logits.detach().float().cpu()); labels.append(y.cpu()); groups.append(g)
        seen += len(x)
    base.synchronize(device)
    stats = {key: value/seen for key, value in sums.items()}
    stats.update(seconds=time.monotonic()-started, samples=seen, optimizer_updates=updates, amp_skipped_updates=skipped)
    return stats, tuple(map(torch.cat, (outputs, labels, groups)))


def report(directory):
    rows = []
    for arm in ARMS:
        path = Path(directory)/arm
        h = json.loads((path/"history.json").read_text())
        best = max(h, key=lambda x: x["validation"]["wga"])
        cfg = json.loads((path/"manifest.json").read_text())["spec"]["config"]
        row = {"arm": arm, "seed": cfg["seed"], "strength": cfg["gradient_intervention"]["strength"],
               "epochs_finished": h[-1]["epoch"], "complete": (path/"training_complete.json").exists(),
               "best_epoch": best["epoch"], "best_val_wga_percent": 100*best["validation"]["wga"],
               "best_val_accuracy_percent": 100*best["validation"]["accuracy"],
               "last_val_wga_percent": 100*h[-1]["validation"]["wga"],
               "training_seconds_including_reference": sum(r["train"]["seconds"] for r in h),
               "mean_residual_energy_fraction": sum(r["train"]["residual_energy_fraction"] for r in h)/len(h)}
        row.update({"val_"+g+"_percent": 100*a for g,a in best["validation"]["group_accuracy"].items()})
        rows.append(row)
    output = Path(directory)/"summary.csv"
    with output.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(output.read_text(), end="")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("train", "report"))
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="data/waterbird_complete95_forest2water2")
    p.add_argument("--teacher", default="runs/teacher_seed0/best.pt")
    p.add_argument("--config", default=str(base.HERE/"config.json"))
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strength", type=float, default=1.)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.command == "report":
        return report(args.out)
    if args.arm is None or args.strength < 0 or not torch.isfinite(torch.tensor(args.strength)):
        p.error("train requires --arm and a finite nonnegative --strength")
    cfg = json.loads(Path(args.config).read_text())
    cfg["seed"] = args.seed
    cfg["gradient_intervention"] = {"arm": args.arm, "strength": args.strength,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference": "FP32 full-token suffix; detached boundary; same CE+KD loss; RNG isolated",
        "controls": "common and within-group cyclic shuffle matched to residual norm per image"}
    base.validate_config(cfg)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    args.method = "tome_kd"
    base.train_epoch = train_epoch
    base.fit(args, cfg, device)


if __name__ == "__main__":
    main()
