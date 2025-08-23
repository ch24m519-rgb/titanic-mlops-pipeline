import os
import shutil
import subprocess
from pathlib import Path
from src.utils import read_params

def download_kaggle(raw_dir):
    comp = "titanic"
    cmd = ["kaggle", "competitions", "download", "-c", comp, "-p", raw_dir]
    subprocess.run(cmd, check=True)
    # unzip all
    for z in Path(raw_dir).glob("*.zip"):
        subprocess.run(["python", "-m", "zipfile", "-e", str(z), raw_dir], check=True)
        z.unlink()

def prepare_local(raw_dir):
    """
    Ensure train.csv is present in raw_dir.
    If it's in project root or elsewhere, copy it automatically.
    """
    os.makedirs(raw_dir, exist_ok=True)

    expected_file = Path(raw_dir) / "train.csv"

    # If already exists, nothing to do
    if expected_file.exists():
        print(f"[INFO] Found {expected_file}")
        return

    # Look for train.csv in project root
    root_file = Path("train.csv")
    if root_file.exists():
        print(f"[INFO] Moving {root_file} -> {expected_file}")
        shutil.move(str(root_file), str(expected_file))
        return

    # Look for train.csv in Downloads (Windows typical case)
    downloads = Path.home() / "Downloads" / "train.csv"
    if downloads.exists():
        print(f"[INFO] Copying {downloads} -> {expected_file}")
        shutil.copy(str(downloads), str(expected_file))
        return

    raise SystemExit(f"Missing train.csv. Please place it in {raw_dir}/ or project root.")

if __name__ == "__main__":
    params = read_params()
    raw_dir = params["data"]["local_raw_dir"]
    source = params["data"]["source"]

    if source == "kaggle":
        if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
            download_kaggle(raw_dir)
        else:
            raise SystemExit("KAGGLE credentials not set. Either set them or switch params.data.source to 'local'")
    else:
        prepare_local(raw_dir)
