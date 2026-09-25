"""Official Waterbirds splits; no groups or segmentation used in training loss."""
import csv
import hashlib
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

GROUPS = ["landbird_land", "landbird_water", "waterbird_land", "waterbird_water"]
EXPECTED_COUNTS = {0: [3498, 184, 56, 1057], 1: [467, 466, 133, 133], 2: [2255, 2255, 642, 642]}


def read_metadata(root):
    root = Path(root).expanduser().resolve()
    with (root / "metadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    seen = set()
    for i, row in enumerate(rows):
        for key in ("y", "place", "split"):
            row[key] = int(row[key])
        if row["y"] not in (0, 1) or row["place"] not in (0, 1) or row["split"] not in (0, 1, 2):
            raise ValueError(f"Invalid labels/split at row {i}")
        p = Path(row["img_filename"])
        if p.is_absolute() or ".." in p.parts or str(p) in seen:
            raise ValueError(f"Duplicate or unsafe image path: {p}")
        seen.add(str(p))
        row["group"] = 2 * row["y"] + row["place"]
        row["sample_id"] = i
    if not rows:
        raise ValueError("Empty metadata")
    return root, rows


def audit(root, verify_images=True, require_official=True):
    root, rows = read_metadata(root)
    counts = {s: [sum(r["split"] == s and r["group"] == g for r in rows) for g in range(4)] for s in range(3)}
    if require_official and counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected official Waterbirds split counts: {counts}")
    if any(min(v) == 0 for v in counts.values()):
        raise ValueError("Each split must contain all four groups")
    digest = hashlib.sha256((root / "metadata.csv").read_bytes()).hexdigest()
    # Hash every encoded image, both to fingerprint the dataset and catch exact duplicates.
    image_digest = hashlib.sha256()
    seen_images = {}
    for row in rows:
        p = root / row["img_filename"]
        if not p.is_file():
            raise FileNotFoundError(p)
        if verify_images:
            encoded = p.read_bytes()
            h = hashlib.sha256(encoded).hexdigest()
            if h in seen_images and seen_images[h] != row["split"]:
                raise ValueError(f"Identical encoded image across splits: {p}")
            seen_images[h] = row["split"]
            image_digest.update(row["img_filename"].encode())
            image_digest.update(bytes.fromhex(h))
            with Image.open(p) as im:
                im.verify()
    return {"root": str(root), "metadata_sha256": digest, "groups": GROUPS,
            "counts": counts, "images_verified": verify_images,
            "images_sha256": image_digest.hexdigest() if verify_images else None}


class Waterbirds(Dataset):
    def __init__(self, root, split, train=False, seed=0):
        self.root, rows = read_metadata(root)
        self.rows = [r for r in rows if r["split"] == split]
        self.seed, self.epoch, self.train = seed, 0, train
        if not self.rows:
            raise ValueError(f"Empty split {split}")
        spatial = [T.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                   T.RandomHorizontalFlip()] if train else [T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224)]
        self.transform = T.Compose(spatial + [T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["img_filename"]) as im:
            im = im.convert("RGB")
            # Match crops/flips by sample and epoch across CE/KD/ToMe, regardless of dropout RNG.
            with torch.random.fork_rng(devices=[]):
                generator = torch.Generator().manual_seed(self.seed + 100003 * self.epoch + 10000019 * row["sample_id"])
                torch.set_rng_state(generator.get_state())
                x = self.transform(im)
        return x, row["y"], row["group"], row["sample_id"]


def loader(dataset, cfg, train=False):
    return DataLoader(dataset, batch_size=cfg["batch_size"] if train else cfg["eval_batch_size"],
                      shuffle=train, num_workers=cfg["workers"], pin_memory=torch.cuda.is_available(), drop_last=False,
                      generator=torch.Generator().manual_seed(cfg["seed"] + 100003 * dataset.epoch))


def metrics(logits, labels, groups, train_weights):
    pred = logits.argmax(1)
    correct = pred.eq(labels)
    counts = [int((groups == g).sum()) for g in range(4)]
    acc = [float(correct[groups == g].float().mean()) if counts[g] else None for g in range(4)]
    valid = all(a is not None for a in acc)
    class_acc = [float(correct[labels == y].float().mean()) if (labels == y).any() else None for y in range(2)]
    return {"accuracy": float(correct.float().mean()), "group_counts": dict(zip(GROUPS, counts)),
            "group_accuracy": dict(zip(GROUPS, acc)), "wga": min(acc) if valid else None,
            "class_macro_accuracy": sum(class_acc) / 2 if None not in class_acc else None,
            "group_macro_accuracy": sum(acc) / 4 if valid else None,
            "train_weighted_accuracy": sum(w*a for w, a in zip(train_weights, acc)) if valid else None}
