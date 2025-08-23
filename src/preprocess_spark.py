import os
from utils import read_params
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml import Pipeline

if __name__ == "__main__":
    params = read_params()
    spark = (SparkSession.builder
             .appName("titanic-preprocess")
             .master(os.getenv("SPARK_MASTER","local[*]"))
             .getOrCreate())

    raw_dir = params["data"]["local_raw_dir"]
    processed_dir = params["data"]["processed_dir"]
    target = params["data"]["target"]
    drop_cols = params["preprocess"]["drop_cols"]

    df = spark.read.csv(f"{raw_dir}/train.csv", header=True, inferSchema=True)

    # Minimal cleaning
    df = df.drop(*[c for c in drop_cols if c in df.columns])

    # Cast target to int
    df = df.withColumn(target, col(target).cast("int"))

    # Create basic features
    # e.g., FamilySize = SibSp + Parch + 1
    if "SibSp" in df.columns and "Parch" in df.columns:
        df = df.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)

    # Impute numerics
    numeric_cols = [c for (c,t) in df.dtypes if t in ("int","double") and c != target]
    imputer = Imputer(strategy="median", inputCols=numeric_cols, outputCols=[f"{c}_imp" for c in numeric_cols])

    # Categorical handling
    cat_cols = [c for (c,t) in df.dtypes if t == "string"]
    indexers = [StringIndexer(handleInvalid="keep", inputCol=c, outputCol=f"{c}_idx") for c in cat_cols]
    encoders = OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols], outputCols=[f"{c}_oh" for c in cat_cols])

    feature_cols = [f"{c}_imp" for c in numeric_cols] + [f"{c}_oh" for c in cat_cols]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    pipeline = Pipeline(stages=[imputer] + indexers + [encoders, assembler])
    model = pipeline.fit(df)
    out = model.transform(df).select("features", target)

    os.makedirs(processed_dir, exist_ok=True)
    out.write.mode("overwrite").parquet(f"{processed_dir}/train.parquet")

    spark.stop()

