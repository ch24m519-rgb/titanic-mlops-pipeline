import os
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from src.utils import read_params

def build_session():
    return (SparkSession.builder
            .appName("titanic-preprocess")
            .master(os.getenv("SPARK_MASTER", "local[*]"))
            .getOrCreate())

if __name__ == "__main__":
    p = read_params()
    spark = build_session()

    raw_path = os.path.join(p["data"]["local_raw_dir"], p["data"]["raw_csv_name"])
    df = (spark.read.option("header", True)
                 .option("inferSchema", True)
                 .csv(raw_path))

    # Select commonly available columns
    cols = ["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"[preprocess] Missing column in CSV: {c}")

    # Basic cleaning / imputation
    df = df.withColumn("Age", df["Age"].cast(DoubleType()))
    df = df.withColumn("Fare", df["Fare"].cast(DoubleType()))
    df = df.withColumn("Survived", df["Survived"].cast("double"))

    # Fill missing numerics with mean
    for c in ["Age", "Fare"]:
        mean_val = df.select(F.mean(F.col(c))).first()[0]
        df = df.fillna({c: float(mean_val) if mean_val is not None else 0.0})

    # Fill Embarked with most frequent
    mode_emb = (df.groupBy("Embarked").count()
                  .orderBy(F.desc("count"))
                  .first())
    df = df.fillna({"Embarked": mode_emb["Embarked"] if mode_emb else "S"})

    # Index categoricals
    si_sex = StringIndexer(inputCol="Sex", outputCol="Sex_index", handleInvalid="keep")
    si_emb = StringIndexer(inputCol="Embarked", outputCol="Embarked_index", handleInvalid="keep")

    df = si_sex.fit(df).transform(df)
    df = si_emb.fit(df).transform(df)

    # Label + features
    label_col = p["data"]["target"]          # "Survived"
    df = df.withColumnRenamed("Survived", label_col)

    feature_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_index", "Embarked_index"]
    va = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
    df = va.transform(df).select("features", label_col)

    # Save as a single file for DVC compatibility
    processed_path = os.path.join(p["data"]["processed_dir"], "titanic.parquet")
    (df.coalesce(1)        # force single output file
       .write.mode("overwrite")
       .parquet(processed_path))

    print(f"[preprocess] Wrote processed dataset to {processed_path}")

    spark.stop()
