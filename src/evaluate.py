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
    
    # ✅ Handle algorithm parameter correctly (deals with duplicate 'train' sections)
    algo_param = p["train"]["algorithm"]
    
    # If algorithm is a list (from second train section), we need to determine 
    # which model was actually saved by trying to load different types
    if isinstance(algo_param, list):
        print(f"[evaluate] Algorithm parameter is a list: {algo_param}")
        print("[evaluate] Attempting to auto-detect saved model type...")
        
        model_path = "models/spark_model"
        model = None
        algo = None
        
        # Try loading as LogisticRegression first (most common)
        try:
            model = LogisticRegressionModel.load(model_path)
            algo = "logreg"
            print("[evaluate] ✅ Successfully loaded LogisticRegression model")
        except Exception as e1:
            print(f"[evaluate] Failed to load as LogisticRegression: {e1}")
            
            # Try loading as RandomForest
            try:
                model = RandomForestClassificationModel.load(model_path)
                algo = "random_forest" 
                print("[evaluate] ✅ Successfully loaded RandomForest model")
            except Exception as e2:
                raise RuntimeError(f"Could not load model as either type. LogReg error: {e1}, RF error: {e2}")
    
    else:
        # Single algorithm specified (string)
        algo = algo_param
        model_path = "models/spark_model"
        
        print(f"[evaluate] Algorithm specified as: {algo}")
        
        if algo == "logreg":
            model = LogisticRegressionModel.load(model_path)
            print("[evaluate] ✅ Loaded LogisticRegression model")
        elif algo == "random_forest":
            model = RandomForestClassificationModel.load(model_path)
            print("[evaluate] ✅ Loaded RandomForest model")
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

    # Load test data
    test = spark.read.parquet(f"{splits_dir}/test.parquet")
    
    # Debug: Check test data structure
    print(f"[evaluate] Test data schema: {test.schema}")
    print(f"[evaluate] Test data count: {test.count()}")
    
    # Make predictions
    preds = model.transform(test)
    
    # Debug: Check predictions distribution to ensure they're not all 0
    print("[evaluate] Checking prediction distribution...")
    pred_dist = preds.groupBy("prediction").count().collect()
    print(f"[evaluate] Prediction distribution: {[(row['prediction'], row['count']) for row in pred_dist]}")
    
    # Check if we have both classes in predictions
    prediction_values = [row['prediction'] for row in pred_dist]
    if len(prediction_values) == 1:
        print(f"[evaluate] ⚠️  WARNING: All predictions are {prediction_values[0]} - model may not be working correctly")
    
    # Evaluate model
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label_col, metricName="f1")
    evaluator_acc = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")
    
    f1 = evaluator_f1.evaluate(preds)
    acc = evaluator_acc.evaluate(preds)

    print(f"[evaluate] Algorithm: {algo}")
    print(f"[evaluate] F1 Score: {f1:.4f}")
    print(f"[evaluate] Accuracy: {acc:.4f}")
    
    # Sanity check for unrealistic results
    if acc == 0.0:
        print("[evaluate] ⚠️  WARNING: Accuracy is 0 - check model and data compatibility")
    elif acc > 0.95:
        print("[evaluate] ⚠️  WARNING: Accuracy > 95% - check for data leakage")
    
    # Log to MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(p["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_param("algorithm", algo)
        mlflow.log_param("model_type_detected", type(model).__name__)
        mlflow.log_metric("f1_eval", f1)
        mlflow.log_metric("accuracy_eval", acc)

    # Write DVC metrics
    with open("metrics.json", "w") as f:
        json.dump({
            "f1": f1, 
            "accuracy": acc,
            "algorithm": algo
        }, f, indent=4)
        
    print(f"[evaluate] Metrics written to metrics.json")

    spark.stop()
