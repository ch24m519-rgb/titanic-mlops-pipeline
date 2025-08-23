import json, subprocess
from src.utils import read_params

if __name__ == "__main__":
    p = read_params()
    if not p["retrain"]["enable"]:
        raise SystemExit("Retrain disabled in params.yaml")
    with open("models/drift_report.json","r") as f:
        rep = json.load(f)
    if rep.get("drift"):
        # Run critical stages again
        subprocess.check_call(["dvc","repro","preprocess"])
        subprocess.check_call(["dvc","repro","split"])
        subprocess.check_call(["dvc","repro","train"])
        subprocess.check_call(["dvc","repro","evaluate"])
        subprocess.check_call(["dvc","repro","register_best"])
    else:
        print("No drift detected; skipping retrain")

