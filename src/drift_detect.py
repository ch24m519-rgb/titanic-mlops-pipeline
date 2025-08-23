import json, os
import numpy as np
from src.utils import read_params, save_json
from pyspark.sql import SparkSession

def psi(expected, actual, bins=10):
    """Population Stability Index for continuous arrays."""
    expected = np.array(expected)
    actual   = np.array(actual)
    q = np.linspace(0,1,bins+1)
    cuts = np.unique(np.quantile(expected, q))
    def bucketize(x, edges):
        return np.clip(np.searchsorted(edges, x, side="right")-1, 0, len(edges)-2)
    e_b = bucketize(expected, cuts)
    a_b = bucketize(actual, cuts)
    psi_val = 0.0
    for b in range(len(cuts)-1):
        e = max((e_b==b).mean(), 1e-6)
        a = max((a_b==b).mean(), 1e-6)
        psi_val += (a - e) * np.log(a / e)
    return psi_val

if __name__ == "__main__":
    p = read_params()
    spark = (SparkSession.builder
             .appName("titanic-drift")
             .master(os.getenv("SPARK_MASTER","local[*]"))
             .getOrCreate())

    processed = p["data"]["processed_dir"]
    df = spark.read.parquet(f"{processed}/train.parquet")
    # In a real system, compare TRAIN vs NEW incoming batch; here we simulate
    data = df.select("features").limit(1000).toPandas()
    base = data["features"].apply(lambda v: float(v[0]))  # project first feature
    new  = data["features"].sample(frac=1.0, random_state=1).apply(lambda v: float(v[0]))

    val = psi(base, new)
    out = {"method": p["drift"]["method"], "psi": float(val), "threshold": p["drift"]["psi_threshold"], "drift": bool(val > p["drift"]["psi_threshold"])}
    save_json(out, "reports/drift_report.json")
    spark.stop()

