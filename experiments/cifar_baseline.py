"""Fixed-endpoint CIFAR-100 CE/KD baseline; no test access during training."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import fcntl
import hashlib
import io
import json
from pathlib import Path
import platform
import random
import sys
import tarfile
import time
import urllib.request

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vendor.mdistiller.resnet import resnet8x4, resnet32x4

UPSTREAM = "a08d46f10d6102bd6e3f258ca5ac880b020ea259"
TEACHER_URL = "https://github.com/megvii-research/mdistiller/releases/download/checkpoints/cifar_teachers.tar"
TEACHER_MEMBER = "cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth"
TEACHER_RAW_SHA256 = "22aa12e2b632b19ddde155ccf2388671842ddd304824b3095e0a7fff364beb4f"
RECIPE = dict(epochs=240, batch_size=64, lr=0.05, momentum=0.9,
              weight_decay=0.0005, milestones=[150, 180, 210], gamma=0.1,
              temperature=4.0, kd_ce_weight=0.1, kd_weight=0.9)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def model_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def configure_device(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use the server's CUDA-enabled .venv.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return device


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def transform(train):
    steps = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if train else []
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(
        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])


def loader(dataset, workers, seed, train):
    return DataLoader(dataset, batch_size=64, shuffle=train, drop_last=False,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed))


def learning_rate(epoch):
    return RECIPE["lr"] * RECIPE["gamma"] ** sum(epoch > m for m in RECIPE["milestones"])


def objective(student, labels, teacher=None):
    ce = F.cross_entropy(student, labels)
    kd = ce.new_zeros(())
    if teacher is None:
        return ce, ce, kd
    t = RECIPE["temperature"]
    kd = F.kl_div(F.log_softmax(student / t, dim=1),
                  F.softmax(teacher / t, dim=1), reduction="batchmean") * t * t
    return RECIPE["kd_ce_weight"] * ce + RECIPE["kd_weight"] * kd, ce, kd


def load_teacher(path, device):
    model = resnet32x4(num_classes=100)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval().requires_grad_(False)


def prepare(args):
    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)
    print("Preparing CIFAR-100 (first download about 169 MB)...", flush=True)
    datasets.CIFAR100(data, train=True, download=True)
    prepare_teacher(data)


def prepare_teacher(data):
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    output = data / "resnet32x4.pt"
    if not output.exists() or not (data / "teacher_provenance.json").exists():
        archive = data / "cifar_teachers.tar"
        if not archive.exists():
            print("Downloading official teacher archive (about 364 MB)...", flush=True)
            tmp = archive.with_suffix(".tar.part")
            urllib.request.urlretrieve(TEACHER_URL, tmp)
            tmp.replace(archive)
        with tarfile.open(archive) as tar:
            # Read the named regular file only; never extract arbitrary archive paths.
            member = tar.getmember(TEACHER_MEMBER)
            if not member.isfile():
                raise ValueError("Teacher archive member is not a regular file")
            raw = tar.extractfile(member).read()
        if hashlib.sha256(raw).hexdigest() != TEACHER_RAW_SHA256:
            raise ValueError("Official teacher differs from the pinned checkpoint; refusing to load")
        numpy_scalar = np.float64(0).__reduce__()[0]
        safe = [argparse.Namespace, (numpy_scalar, "numpy.core.multiarray.scalar"),
                np.dtype, type(np.dtype("float64")), type(np.dtype("float32"))]
        with torch.serialization.safe_globals(safe):
            checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        state = checkpoint["model"]
        model = resnet32x4(num_classes=100)
        model.load_state_dict(state, strict=True)
        atomic_save(output, model.state_dict())
        atomic_json(data / "teacher_provenance.json", dict(url=TEACHER_URL,
            upstream_commit=UPSTREAM, member=TEACHER_MEMBER, archive_sha256=sha(archive),
            raw_checkpoint_sha256=hashlib.sha256(raw).hexdigest(), converted_sha256=sha(output),
            teacher_model_hash=model_hash(model), setup_seconds=time.monotonic()-started))
    provenance = json.loads((data / "teacher_provenance.json").read_text())
    if sha(output) != provenance["converted_sha256"]:
        raise ValueError("Teacher checkpoint changed since preparation")
    print(f"Prepared official teacher: {output}", flush=True)


@torch.inference_mode()
def evaluate(model, batches, device, limit=None):
    model.eval()
    logits, labels = [], []
    for i, (x, y) in enumerate(batches):
        if limit is not None and i >= limit:
            break
        z, _ = model(x.to(device, non_blocking=True))
        if not torch.isfinite(z).all():
            raise FloatingPointError("Nonfinite evaluation logits")
        logits.append(z.cpu()); labels.append(y)
    z, y = torch.cat(logits), torch.cat(labels)
    metrics = dict(samples=len(y), accuracy_percent=100 * float((z.argmax(1) == y).float().mean()),
                   ce=float(F.cross_entropy(z, y)))
    return metrics, dict(logits=z, labels=y)


def preflight(args):
    device = configure_device(args.device)
    data = datasets.CIFAR100(args.data, train=True, transform=transform(False))
    teacher = load_teacher(Path(args.data) / "resnet32x4.pt", device)
    before = model_hash(teacher)
    metrics, _ = evaluate(teacher, loader(data, args.workers, 0, False), device, limit=16)
    seed_all(0)
    student = resnet8x4(num_classes=100).to(device).train()
    x, y = next(iter(loader(data, 0, 0, False)))
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        target, _ = teacher(x)
    z, _ = student(x)
    loss, _, _ = objective(z, y, target)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                     for p in student.parameters()):
        raise FloatingPointError("Preflight KD gradients are nonfinite")
    if before != model_hash(teacher):
        raise RuntimeError("Frozen teacher changed during preflight")
    if metrics["accuracy_percent"] < 60:
        raise RuntimeError(f"Teacher train diagnostic unexpectedly low: {metrics}. Check checkpoint/data.")
    print("Teacher TRAIN diagnostic (not validation/test): " + json.dumps(metrics), flush=True)
    print("KD forward/backward and frozen-teacher checks passed.", flush=True)


def spec_for(args):
    data = Path(args.data)
    return dict(recipe=RECIPE, method=args.method, seed=args.seed, workers=args.workers,
        device=args.device, precision="fp32_no_tf32", train_samples=50000,
        train_sha256=sha(data / "cifar-100-python/train"),
        teacher_sha256=sha(data / "resnet32x4.pt"),
        source_sha256=sha(__file__), model_source_sha256=sha(ROOT / "vendor/mdistiller/resnet.py"),
        torch=str(torch.__version__), torchvision=str(torchvision.__version__),
        numpy=np.__version__, python=platform.python_version(), upstream_commit=UPSTREAM)


def train_epoch(student, teacher, dataset, optimizer, seed, epoch, workers, device):
    epoch_seed = seed + 100003 * epoch
    seed_all(epoch_seed)
    student.train()
    if teacher is not None:
        teacher.eval()
    lr = learning_rate(epoch)
    for group in optimizer.param_groups:
        group["lr"] = lr
    sync(device)
    started = time.monotonic()
    seen = 0
    correct = torch.zeros((), device=device, dtype=torch.int64)
    losses = torch.zeros(3, device=device, dtype=torch.float64)
    trace = hashlib.sha256()
    first_batch_sha = None
    for x, y in loader(dataset, workers, epoch_seed, True):
        if first_batch_sha is None:
            first_batch_sha = hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
        trace.update(y.numpy().tobytes())
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        target = None
        if teacher is not None:
            with torch.no_grad():
                target, _ = teacher(x)
        optimizer.zero_grad(set_to_none=True)
        z, _ = student(x)
        loss, ce, kd = objective(z, y, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        losses += torch.stack([loss.detach(), ce.detach(), kd.detach()]).double() * len(y)
        correct += (z.argmax(1) == y).sum()
        seen += len(y)
    sync(device)
    return dict(epoch=epoch, lr=lr, **{k: v/seen for k, v in zip(("loss", "ce", "kd"), losses.tolist())},
        train_accuracy_percent=100*int(correct)/seen, samples=seen,
        teacher_images=seen if teacher is not None else 0,
        seconds=time.monotonic()-started, first_batch_sha256=first_batch_sha,
        label_order_sha256=trace.hexdigest())


def fit(args):
    device = configure_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = spec_for(args)
        manifest = out / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text())["spec"] != spec:
                raise ValueError("Run code/config/data/environment changed. Use a new output directory.")
            if not args.resume:
                raise ValueError("Run exists; pass --resume or choose a new directory.")
        else:
            atomic_json(manifest, dict(spec=spec, endpoint="fixed epoch 240; no test-based selection",
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                cuda_runtime=torch.version.cuda, created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
        if (out / "training_complete.json").exists():
            done = json.loads((out / "training_complete.json").read_text())
            if sha(out / "last.pt") != done["checkpoint_sha256"]:
                raise ValueError("Completed checkpoint changed")
            print(f"Already complete: {out}", flush=True)
            return
        dataset = datasets.CIFAR100(args.data, train=True, transform=transform(True))
        seed_all(args.seed)
        student = resnet8x4(num_classes=100).to(device)
        initial_hash = model_hash(student)
        teacher = load_teacher(Path(args.data)/"resnet32x4.pt", device) if args.method == "kd" else None
        optimizer = torch.optim.SGD(student.parameters(), lr=RECIPE["lr"],
            momentum=RECIPE["momentum"], weight_decay=RECIPE["weight_decay"])
        history, prior_wall = [], 0.
        if (out / "last.pt").exists():
            state = torch.load(out / "last.pt", map_location=device, weights_only=True)
            if state["spec"] != spec or state["initial_model_sha256"] != initial_hash:
                raise ValueError("Checkpoint provenance mismatch")
            student.load_state_dict(state["student"])
            optimizer.load_state_dict(state["optimizer"])
            history, prior_wall = state["history"], state["wall_seconds_before_save"]
        start = time.monotonic()
        for epoch in range(len(history)+1, RECIPE["epochs"]+1):
            row = train_epoch(student, teacher, dataset, optimizer, args.seed, epoch, args.workers, device)
            history.append(row)
            atomic_save(out / "last.pt", dict(student=student.state_dict(), optimizer=optimizer.state_dict(),
                history=history, spec=spec, initial_model_sha256=initial_hash,
                wall_seconds_before_save=prior_wall+time.monotonic()-start))
            atomic_json(out / "history.json", history)
            print(f"{args.method} seed={args.seed} epoch={epoch}/{RECIPE['epochs']} loss={row['loss']:.4f} "
                  f"train_acc={row['train_accuracy_percent']:.2f}% lr={row['lr']:.5g} "
                  f"time={row['seconds']:.1f}s", flush=True)
        atomic_json(out / "training_complete.json", dict(epochs=RECIPE["epochs"], initial_model_sha256=initial_hash,
            training_seconds=sum(r["seconds"] for r in history),
            wall_seconds=prior_wall+time.monotonic()-start,
            teacher_images=sum(r["teacher_images"] for r in history), checkpoint_sha256=sha(out/"last.pt")))
        print(f"[TRAIN DONE] {out}; test has not been evaluated.", flush=True)


def finish(args):
    device = configure_device(args.device)
    runs = Path(args.runs)
    report = runs / f"report_seed{args.seed}"
    report.mkdir(parents=True, exist_ok=True)
    with (report / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        states, hashes = {}, {}
        for method in ("ce", "kd"):
            path = runs / f"{method}_seed{args.seed}"
            done = json.loads((path / "training_complete.json").read_text())
            hashes[method] = sha(path / "last.pt")
            if done["epochs"] != RECIPE["epochs"] or hashes[method] != done["checkpoint_sha256"]:
                raise ValueError("Both fixed-endpoint runs must be complete and unchanged")
            state = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
            if state["spec"]["method"] != method or state["spec"]["seed"] != args.seed:
                raise ValueError("Checkpoint method/seed does not match its run directory")
            if state["spec"]["recipe"] != RECIPE:
                raise ValueError("Checkpoint does not use the precommitted recipe")
            if len(state["history"]) != RECIPE["epochs"]:
                raise ValueError("Incomplete checkpoint")
            states[method] = state
        ce, kd = states["ce"], states["kd"]
        if ce["initial_model_sha256"] != kd["initial_model_sha256"]:
            raise ValueError("Unpaired student initialization")
        for key in ce["spec"]:
            if key != "method" and ce["spec"][key] != kd["spec"][key]:
                raise ValueError(f"Unpaired run setting: {key}")
        for a, b in zip(ce["history"], kd["history"]):
            for key in ("first_batch_sha256", "label_order_sha256"):
                if a[key] != b[key]:
                    raise ValueError(f"Unpaired training data at epoch {a['epoch']}: {key}")
        if sha(Path(args.data)/"resnet32x4.pt") != ce["spec"]["teacher_sha256"]:
            raise ValueError("Teacher changed")
        if sha(Path(args.data)/"cifar-100-python/train") != ce["spec"]["train_sha256"]:
            raise ValueError("Training dataset changed")
        if (report / "results.json").exists():
            previous = json.loads((report / "results.json").read_text())
            if previous["checkpoint_sha256"] != hashes:
                raise ValueError("Report exists for different checkpoints")
            print((report / "summary.csv").read_text(), end="")
            return
        dataset = datasets.CIFAR100(args.data, train=False, transform=transform(False))
        rows = []
        for method in ("teacher", "ce", "kd"):
            if method == "teacher":
                model = load_teacher(Path(args.data)/"resnet32x4.pt", device)
            else:
                model = resnet8x4(num_classes=100).to(device)
                model.load_state_dict(states[method]["student"])
            metrics, predictions = evaluate(model, loader(dataset, args.workers, 0, False), device)
            atomic_save(report / f"{method}_test_predictions.pt", predictions)
            history = states[method]["history"] if method != "teacher" else []
            rows.append(dict(method=method, seed=args.seed, epoch=RECIPE["epochs"], split="test", **metrics,
                student_training_seconds=sum(r["seconds"] for r in history) if history else None,
                teacher_images=sum(r["teacher_images"] for r in history) if history else None))
            del model
        delta = rows[2]["accuracy_percent"] - rows[1]["accuracy_percent"]
        tmp = report / "summary.csv.tmp"
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        tmp.replace(report / "summary.csv")
        atomic_json(report / "results.json", dict(records=rows, checkpoint_sha256=hashes,
            test_sha256=sha(Path(args.data)/"cifar-100-python/test"), kd_minus_ce_pp=delta,
            interpretation="Single-seed fixed-endpoint baseline, not a new method or proof of KD benefit."))
        print((report / "summary.csv").read_text(), end="")
        print(f"KD - CE = {delta:+.3f} percentage points (seed {args.seed}; no multi-seed conclusion).")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "preflight", "train", "finish"])
    p.add_argument("--data", default="data/cifar100")
    p.add_argument("--runs", default="runs/cifar100_baseline")
    p.add_argument("--out")
    p.add_argument("--method", choices=["ce", "kd"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if args.workers < 0 or args.seed < 0:
        p.error("workers and seed must be nonnegative")
    if args.command == "train" and (not args.out or not args.method):
        p.error("train requires --out and --method")
    {"prepare": prepare, "preflight": preflight, "train": fit, "finish": finish}[args.command](args)


if __name__ == "__main__":
    main()
