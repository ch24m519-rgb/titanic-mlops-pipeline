import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from src.utils import read_params

def build_session():
    return (SparkSession.builder
            .appName("titanic-split")
            .master(os.getenv("SPARK_MASTER", "local[*]"))
            .getOrCreate())

if __name__ == "__main__":
    p = read_params()
    spark = build_session()

    # ✅ Fixed: directly use processed_dir + fixed filename
    processed = os.path.join(p["data"]["processed_dir"], "titanic.parquet")
    splits_dir = p["data"]["splits_dir"]
    test_size = p["split"]["test_size"]
    random_state = p["split"]["random_state"]

    # Read processed dataset
    df = spark.read.parquet(processed)

    # Add random column for splitting
    df = df.withColumn("rand", rand(seed=random_state))

    train_df = df.filter(df.rand >= test_size).drop("rand")
    test_df = df.filter(df.rand < test_size).drop("rand")

    # Save splits
    train_out = os.path.join(splits_dir, "train.parquet")
    test_out = os.path.join(splits_dir, "test.parquet")

    (train_df.coalesce(1).write.mode("overwrite").parquet(train_out))
    (test_df.coalesce(1).write.mode("overwrite").parquet(test_out))

    print(f"[split] Wrote train={train_out}, test={test_out}")

    spark.stop()
