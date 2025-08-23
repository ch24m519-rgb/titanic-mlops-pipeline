import os, mlflow
from utils import read_params
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == "__main__":
    p = read_params()
    spark = (SparkSession.builder
             .appName("titanic-train")
             .master(os.getenv("SPARK_MASTER","local[*]"))
             .getOrCreate())

    mlflow.set_experiment(p["mlflow"]["experiment_name"])
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000"))

    train = spark.read.parquet(f"{p['data']['splits_dir']}/train.parquet")
    test  = spark.read.parquet(f"{p['data']['splits_dir']}/test.parquet")

    algo = p["train"]["algorithm"]
    with mlflow.start_run():
        if algo=="logreg":
            clf = LogisticRegression(featuresCol="features", labelCol=p["data"]["target"],
                                     maxIter=p["train"]["max_iter"],
                                     regParam=p["train"]["reg_param"],
                                     elasticNetParam=p["train"]["elastic_net_param"])
        else:
            clf = RandomForestClassifier(featuresCol="features", labelCol=p["data"]["target"],
                                         numTrees=p["train"]["rf_trees"],
                                         maxDepth=p["train"]["rf_max_depth"])

        model = clf.fit(train)
        preds = model.transform(test)

        evaluator_f1 = MulticlassClassificationEvaluator(labelCol=p["data"]["target"], metricName="f1")
        evaluator_acc = MulticlassClassificationEvaluator(labelCol=p["data"]["target"], metricName="accuracy")
        f1 = evaluator_f1.evaluate(preds)
        acc = evaluator_acc.evaluate(preds)

        mlflow.log_param("algorithm", algo)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", acc)

        # Log Spark ML model as a pyfunc using mlflow's spark flavor
        mlflow.spark.log_model(model, artifact_path="model", registered_model_name=None)

        # Save locally for evaluate stage
        os.makedirs("models", exist_ok=True)
        mlflow.spark.save_model(model, "models/spark_model")

    spark.stop()

