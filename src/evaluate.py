import json
from utils import read_params, save_json
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator

if __name__ == "__main__":
    p = read_params()
    spark = (SparkSession.builder
             .appName("titanic-eval")
             .master("local[*]")
             .getOrCreate())

    target = p["data"]["target"]
    test  = spark.read.parquet(f"{p['data']['splits_dir']}/test.parquet")

    from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
    # Load model saved by train stage
    try:
        model = LogisticRegressionModel.load("models/spark_model")
    except Exception:
        model = RandomForestClassificationModel.load("models/spark_model")

    preds = model.transform(test)
    evaluator = BinaryClassificationEvaluator(labelCol=target, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(preds)
    save_json({"auc": auc}, "models/metrics.json")
    spark.stop()

