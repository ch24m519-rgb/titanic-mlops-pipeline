import os
import json
import mlflow
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
from src.utils import read_params

def build_session():
    return (SparkSession.builder
            .appName("titanic-evaluate")
            .master(os.getenv("SPARK_MASTER", "local[*]"))
            .getOrCreate())

if __name__ == "__main__":
    p = read_params()
    spark = build_session()

    # Inputs
    splits_dir = p["data"]["splits_dir"]
    label_col = p["data"]["target"]
    algo = p["train"]["algorithm"]

    test = spark.read.parquet(f"{splits_dir}/test.parquet")

    # Load model saved by train stage
    model_path = "models/spark_model"
    if algo == "logreg":
        model = LogisticRegressionModel.load(model_path)
    else:
        model = RandomForestClassificationModel.load(model_path)

    preds = model.transform(test)

    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label_col, metricName="f1")
    evaluator_acc = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")
    f1 = evaluator_f1.evaluate(preds)
    acc = evaluator_acc.evaluate(preds)

    print(f"[evaluate] Algo={algo}, f1={f1:.4f}, acc={acc:.4f}")

    # Log to MLflow (optional but nice)
    mlflow.set_tracking_uri(os.getenv(p["mlflow"]["tracking_uri_env"], "http://localhost:5000"))
    mlflow.set_experiment(p["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_param("algorithm", algo)
        mlflow.log_metric("f1_eval", f1)
        mlflow.log_metric("accuracy_eval", acc)

    # Write DVC metrics
    with open("metrics.json", "w") as f:
        json.dump({"f1": f1, "accuracy": acc}, f, indent=4)

    spark.stop()
