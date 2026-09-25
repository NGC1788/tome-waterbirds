"""Download the official Waterbirds dataset (no synthetic reconstruction)."""
import argparse
from pathlib import Path
import subprocess
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", default="data")
    args = p.parse_args()
    destination = Path(args.destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination/"waterbird_complete95_forest2water2.tar.gz"
    root = destination/"waterbird_complete95_forest2water2"
    if (root/"metadata.csv").is_file():
        print(f"Existing dataset retained: {root}")
        return
    subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-",
                    "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz",
                    "--output", str(archive)], check=True)
    with tarfile.open(archive) as f:
        f.extractall(destination, filter="data")
    if not (root/"metadata.csv").is_file():
        raise RuntimeError("Expected official metadata.csv not found after extraction")
    print(f"Dataset: {root}. Run preflight before training.")


if __name__ == "__main__":
    main()
