import os
from src.utils import read_params
from pyspark.sql import SparkSession

if __name__ == "__main__":
    p = read_params()
    spark = (SparkSession.builder
             .appName("titanic-split")
             .master(os.getenv("SPARK_MASTER","local[*]"))
             .getOrCreate())

    processed = p["data"]["processed_dir"]
    splits_dir = p["data"]["splits_dir"]
    test_size = p["split"]["test_size"]
    seed = p["split"]["random_state"]

    df = spark.read.parquet(f"{processed}/train.parquet")
    train_df, test_df = df.randomSplit([1.0 - test_size, test_size], seed=seed)

    os.makedirs(splits_dir, exist_ok=True)
    train_df.write.mode("overwrite").parquet(f"{splits_dir}/train.parquet")
    test_df.write.mode("overwrite").parquet(f"{splits_dir}/test.parquet")
    spark.stop()

