import os
import shutil
from pathlib import Path
from src.utils import read_params  # assumes you already have this

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    p = read_params()
    raw_name = p["data"]["raw_csv_name"]
    dst = Path("data/raw") / raw_name
    ensure_dir("data/raw")

    # Priority: already at destination → do nothing
    if dst.exists():
        print(f"[get_data] Found {dst}, nothing to do.")
        raise SystemExit(0)

    # Otherwise, try common local sources
    candidates = [
        Path("data/raw/train.csv"),
        Path("train.csv"),
        Path("data/raw/titanic.csv"),  # in case user already created it
    ]
    for c in candidates:
        if c.exists():
            shutil.copy2(c, dst)
            print(f"[get_data] Copied {c} -> {dst}")
            break
    else:
        raise FileNotFoundError(
            "[get_data] Could not find a local Titanic CSV. "
            "Put train.csv in project root or data/raw/, then rerun."
        )
