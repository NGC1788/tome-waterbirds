"""Phase 1: preflight, teacher/CE/KD/ToMe-KD training, then explicit final evaluation."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from contextlib import nullcontext
import hashlib
import fcntl
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import timm
import torch
import torchvision
import PIL
from torch.nn import functional as F

from data import Waterbirds, audit, loader, metrics
from models import build, URLS

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hash():
    h = hashlib.sha256()
    for p in sorted([*HERE.glob("*.py"), *HERE.glob("vendor/**/*.py")]):
        h.update(str(p.relative_to(HERE)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def save(path, state):
    temp = Path(path).with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__), "timm": timm.__version__,
            "torchvision": str(torchvision.__version__), "numpy": np.__version__, "pillow": PIL.__version__,
            "platform": platform.platform(), "device": str(device), "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "source_sha256": source_hash()}


def validate_config(cfg):
    for key in ("batch_size", "eval_batch_size", "accumulation_steps", "teacher_epochs", "student_epochs", "train_eval_every"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["workers"] < 0 or cfg["temperature"] <= 0 or not 0 <= cfg["kd_alpha"] <= 1:
        raise ValueError("Invalid worker count, temperature or KD coefficient")
    if cfg["lr"] <= 0 or not 0 <= cfg["min_lr"] <= cfg["lr"]:
        raise ValueError("Invalid learning rate")


def loss_fn(logits, labels, targets, cfg):
    ce = F.cross_entropy(logits.float(), labels, label_smoothing=cfg["label_smoothing"])
    if targets is None:
        return ce, ce.detach(), ce.detach().new_zeros(())
    t = cfg["temperature"]
    kd = F.kl_div(F.log_softmax(logits.float()/t, -1), F.softmax(targets.float()/t, -1), reduction="batchmean") * t*t
    return (1-cfg["kd_alpha"])*ce + cfg["kd_alpha"]*kd, ce.detach(), kd.detach()


def autocast(cfg, device):
    return torch.autocast("cuda", dtype=torch.float16) if cfg["amp"] and device.type == "cuda" else nullcontext()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate(model, dataset, cfg, device, weights, merge=False, return_predictions=False):
    model.eval()
    outputs, labels, groups, ids = [], [], [], []
    for x, y, g, sid in loader(dataset, cfg):
        with autocast(cfg, device):
            logits = model(x.to(device, non_blocking=True), merge=merge)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation output")
        outputs.append(logits.float().cpu())
        labels.append(y); groups.append(g); ids.append(sid)
    outputs, labels, groups, ids = map(torch.cat, (outputs, labels, groups, ids))
    result = metrics(outputs, labels, groups, weights)
    result["ce"] = float(F.cross_entropy(outputs, labels))
    return (result, {"logits": outputs, "labels": labels, "groups": groups, "sample_ids": ids}) if return_predictions else result


def epoch_lr(cfg, epoch, total):
    warmup = min(cfg["warmup_epochs"], max(0, total-1))
    if epoch <= warmup:
        return cfg["lr"] * epoch / warmup
    progress = (epoch-warmup-1) / max(1, total-warmup-1)
    return cfg["min_lr"] + (cfg["lr"]-cfg["min_lr"])*(1+math.cos(math.pi*progress))/2


def optimizer_for(model, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        group = no_decay if p.ndim < 2 or name.endswith(("pos_embed", "cls_token")) else decay
        group.append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"])


def train_epoch(model, teacher, dataset, cfg, device, optimizer, scaler, epoch, merge):
    seed_all(cfg["seed"] + epoch*100003)
    dataset.epoch = epoch
    batches = loader(dataset, cfg, train=True)
    model.train()
    if teacher is not None:
        teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    outputs, labels, groups = [], [], []
    sums = dict(loss=0., ce=0., kd=0.)
    skipped, seen, updates = 0, 0, 0
    synchronize(device)
    started = time.monotonic()
    for step, (x, y, g, _) in enumerate(batches):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(cfg, device):
            targets = None
            if teacher is not None:
                with torch.no_grad():
                    targets = teacher(x, merge=False)
            logits = model(x, merge=merge)
            loss, ce, kd = loss_fn(logits, y, targets, cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at {epoch=} {step=}")
        first = step // cfg["accumulation_steps"] * cfg["accumulation_steps"]
        last = min(first+cfg["accumulation_steps"], len(batches))
        samples_in_window = min(last*cfg["batch_size"], len(dataset)) - first*cfg["batch_size"]
        scaler.scale(loss * len(x)/samples_in_window).backward()
        if step+1 == last:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=not scaler.is_enabled())
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skip = scaler.get_scale() < scale
            skipped += int(skip); updates += int(not skip)
            optimizer.zero_grad(set_to_none=True)
        for key, value in (("loss", loss), ("ce", ce), ("kd", kd)):
            sums[key] += float(value.detach())*len(x)
        outputs.append(logits.detach().float().cpu()); labels.append(y.cpu()); groups.append(g)
        seen += len(x)
    synchronize(device)
    stats = {key: value/seen for key, value in sums.items()}
    stats.update(seconds=time.monotonic()-started, samples=seen, optimizer_updates=updates, amp_skipped_updates=skipped)
    return stats, tuple(map(torch.cat, (outputs, labels, groups)))


def fit(args, cfg, device):
    directory = Path(args.out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    run_lock = (directory/".lock").open("a")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another process is writing this run") from None
    role = "teacher" if args.method == "teacher" else "student"
    merge = args.method == "tome_kd"
    use_kd = args.method in ("kd", "tome_kd")
    if use_kd and not args.teacher:
        raise ValueError("--teacher is required for KD")
    data_info = audit(args.data)
    teacher_hash = sha(args.teacher) if use_kd else None
    spec = {"config": cfg, "method": args.method, "role": role, "initialization": "imagenet",
            "metadata_sha256": data_info["metadata_sha256"], "images_sha256": data_info["images_sha256"],
            "teacher_sha256": teacher_hash, "source_sha256": source_hash(),
            "runtime_versions": {"torch": str(torch.__version__), "timm": timm.__version__}}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["spec"] != spec:
            raise ValueError("Run config/data/code/teacher changed. Use a new output directory.")
        if not args.resume:
            raise ValueError("Run exists. Specify --resume or use a new directory.")
    elif args.resume:
        raise ValueError("No run to resume")
    else:
        write_json(manifest_path, {"spec": spec, "data": data_info, "environment": environment(device),
                                  "pretrained_url": URLS[role], "selection": "highest validation WGA, earliest tie",
                                  "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    state = torch.load(directory/"last.pt", map_location="cpu", weights_only=True) if args.resume and (directory/"last.pt").exists() else None
    seed_all(cfg["seed"])
    model = build(role, cfg, pretrained=state is None).to(device)
    teacher = None
    if use_kd:
        teacher_state = torch.load(args.teacher, map_location="cpu", weights_only=True)
        ts = teacher_state["spec"]
        if ts["role"] != "teacher" or ts["metadata_sha256"] != spec["metadata_sha256"] or ts["images_sha256"] != spec["images_sha256"]:
            raise ValueError("Teacher role/data mismatch")
        teacher = build("teacher", ts["config"], pretrained=False).to(device)
        teacher.load_state_dict(teacher_state["model"])
        teacher.requires_grad_(False).eval()
    train = Waterbirds(args.data, 0, train=True, seed=cfg["seed"])
    train_eval, val = Waterbirds(args.data, 0), Waterbirds(args.data, 1)
    weights = [n/len(train) for n in data_info["counts"][0]]
    optimizer = optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")
    start, best_score, best_epoch, history = 0, -1., None, []
    if state:
        if state["spec"] != spec:
            raise ValueError("Checkpoint fingerprint mismatch")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, best_score, best_epoch, history = state["epoch"], state["best_score"], state["best_epoch"], state["history"]
        del state
    total = cfg["teacher_epochs"] if role == "teacher" else cfg["student_epochs"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start+1, total+1):
        began = time.monotonic()
        lr = epoch_lr(cfg, epoch, total)
        for group in optimizer.param_groups:
            group["lr"] = lr
        stats, online = train_epoch(model, teacher, train, cfg, device, optimizer, scaler, epoch, merge)
        stats["online_accuracy"] = metrics(*online, weights)
        validation = evaluate(model, val, cfg, device, weights, merge)
        if validation["wga"] is None:
            raise ValueError("Validation is missing a group")
        record = {"epoch": epoch, "lr": lr, "train": stats, "validation": validation}
        if epoch == 1 or epoch % cfg["train_eval_every"] == 0 or epoch == total:
            record["train_eval"] = evaluate(model, train_eval, cfg, device, weights, merge)
        record["epoch_wall_seconds"] = time.monotonic()-began
        record["peak_gpu_gib"] = torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else None
        payload = {"model": model.state_dict(), "spec": spec, "epoch": epoch, "validation": validation, "train_weights": weights}
        if validation["wga"] > best_score:
            best_score, best_epoch = validation["wga"], epoch
            save(directory/"best.pt", payload)
        history.append(record)
        if epoch in (1, 10, 30, 60, 100):
            save(directory/f"epoch_{epoch:03d}.pt", payload)
        save(directory/"last.pt", {**payload, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                                  "best_score": best_score, "best_epoch": best_epoch, "history": history})
        write_json(directory/"history.json", history)
        print(f"{args.method} epoch={epoch}/{total} loss={stats['loss']:.4f} val_wga={validation['wga']:.4f} time={record['epoch_wall_seconds']:.1f}s", flush=True)
    write_json(directory/"training_complete.json", {"best_epoch": best_epoch, "best_validation_wga": best_score,
                                                   "epochs": total, "test_evaluated": False})
    run_lock.close()


def final_evaluation(args, device):
    path = Path(args.checkpoint).resolve()
    state = torch.load(path, map_location="cpu", weights_only=True)
    spec, cfg = state["spec"], state["spec"]["config"]
    info = audit(args.data)
    if info["metadata_sha256"] != spec["metadata_sha256"] or info["images_sha256"] != spec["images_sha256"]:
        raise ValueError("Evaluation data mismatch")
    if spec["source_sha256"] != source_hash():
        raise ValueError("Evaluation code differs from training code")
    model = build(spec["role"], cfg, pretrained=False).to(device)
    model.load_state_dict(state["model"])
    test = Waterbirds(args.data, 2)
    results = {}
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    for merge in ([False] if spec["role"] == "teacher" else [False, True]):
        name = "merge_on" if merge else "merge_off"
        result, predictions = evaluate(model, test, cfg, device, state["train_weights"], merge, True)
        save(output/f"{name}_predictions.pt", predictions)
        results[name] = result
    write_json(output/"test_results.json", {"checkpoint": str(path), "checkpoint_sha256": sha(path),
                                         "epoch": state["epoch"], "spec": spec, "metrics": results})
    print(json.dumps(results, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("preflight", "train", "evaluate"))
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--config", default=str(HERE/"config.json"))
    p.add_argument("--seed", type=int)
    p.add_argument("--method", choices=("teacher", "ce", "kd", "tome_kd"))
    p.add_argument("--teacher")
    p.add_argument("--checkpoint")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    if args.command == "preflight":
        write_json(args.out, {"data": audit(args.data), "environment": environment(device)})
        print(f"Preflight passed: {args.out}")
    elif args.command == "train":
        if not args.method:
            p.error("train requires --method")
        cfg = json.loads(Path(args.config).read_text())
        if args.seed is not None:
            cfg["seed"] = args.seed
        validate_config(cfg)
        fit(args, cfg, device)
    else:
        if not args.checkpoint:
            p.error("evaluate requires --checkpoint")
        final_evaluation(args, device)


if __name__ == "__main__":
    main()
