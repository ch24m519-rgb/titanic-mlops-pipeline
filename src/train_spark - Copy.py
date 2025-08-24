import os
import mlflow
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from src.utils import read_params

def build_session():
    return (SparkSession.builder
            .appName("titanic-train")
            .master(os.getenv("SPARK_MASTER", "local[*]"))
            .getOrCreate())

if __name__ == "__main__":
    p = read_params()
    spark = build_session()

    # MLflow setup
    mlflow.set_tracking_uri(os.getenv(p["mlflow"]["tracking_uri_env"], "http://localhost:5000"))
    mlflow.set_experiment(p["mlflow"]["experiment_name"])

    # Data
    splits_dir = p["data"]["splits_dir"]
    label_col = p["data"]["target"]
    train = spark.read.parquet(f"{splits_dir}/train.parquet")
    test  = spark.read.parquet(f"{splits_dir}/test.parquet")

    algo = p["train"]["algorithm"]

    with mlflow.start_run(run_name="train"):
        if algo == "logreg":
            clf = LogisticRegression(
                featuresCol="features", labelCol=label_col,
                maxIter=p["train"]["max_iter"],
                regParam=p["train"]["reg_param"],
                elasticNetParam=p["train"]["elastic_net_param"]
            )
        else:
            clf = RandomForestClassifier(
                featuresCol="features", labelCol=label_col,
                numTrees=p["train"]["rf_trees"],
                maxDepth=p["train"]["rf_max_depth"]
            )

        model = clf.fit(train)
        preds = model.transform(test)

        evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label_col, metricName="f1")
        evaluator_acc = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")
        f1 = evaluator_f1.evaluate(preds)
        acc = evaluator_acc.evaluate(preds)

        print(f"[train] Algo={algo}, f1={f1:.4f}, acc={acc:.4f}")

        mlflow.log_param("algorithm", algo)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", acc)

        os.makedirs("models", exist_ok=True)
        model.save("models/spark_model")  # Spark-native format

    spark.stop()
