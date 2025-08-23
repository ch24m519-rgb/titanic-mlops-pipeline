import os
import shutil
import subprocess
from pathlib import Path
from utils import read_params

def download_kaggle(raw_dir):
    # Requires KAGGLE_USERNAME & KAGGLE_KEY set; else skip
    comp = "titanic"
    cmd = ["kaggle", "competitions", "download", "-c", comp, "-p", raw_dir]
    try:
        subprocess.run(cmd, check=True)
        # unzip all
        for z in Path(raw_dir).glob("*.zip"):
            subprocess.run(["python","-m","zipfile","-e",str(z),raw_dir], check=True)
            z.unlink()
    except Exception as e:
        raise RuntimeError(f"Kaggle download failed. Ensure credentials. Error: {e}")

if __name__ == "__main__":
    params = read_params()
    raw_dir = params["data"]["local_raw_dir"]

    os.makedirs(raw_dir, exist_ok=True)
    source = params["data"]["source"]
    if source == "kaggle":
        if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
            download_kaggle(raw_dir)
        else:
            raise SystemExit("KAGGLE credentials not set. Either set them or switch params.data.source to 'local' and place CSVs in data/raw")
    else:
        # expect user provided data/raw/train.csv & test.csv
        required = ["train.csv"]
        for f in required:
            if not Path(raw_dir, f).exists():
                raise SystemExit(f"Missing {f} in {raw_dir}")
