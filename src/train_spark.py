import os
import mlflow

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

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
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(p["mlflow"]["experiment_name"])

    # Data
    splits_dir = p["data"]["splits_dir"]
    label_col = p["data"]["target"]
    train = spark.read.parquet(os.path.join(splits_dir, "train.parquet"))
    test = spark.read.parquet(os.path.join(splits_dir, "test.parquet"))
    
    # Get algorithm(s) from params - handle both single string and list
    algo_param = p["train"]["algorithm"]
    if isinstance(algo_param, list):
        algorithms = algo_param
    else:
        algorithms = [algo_param]

    with mlflow.start_run(run_name="train_with_tuning") as parent_run:
        best_overall_f1 = 0
        best_overall_model = None
        best_algo = None
        
        for algo in algorithms:
            print(f"\n[Training] Algorithm: {algo}")
            print("=" * 50)
            
            # Build parameter grid from params.yaml
            if algo == "logreg":
                clf = LogisticRegression(featuresCol="features", labelCol=label_col)
                
                # Get hyperparameter lists from params.yaml
                max_iters = p["train"]["max_iter"] if isinstance(p["train"]["max_iter"], list) else [p["train"]["max_iter"]]
                reg_params = p["train"]["reg_param"] if isinstance(p["train"]["reg_param"], list) else [p["train"]["reg_param"]]
                elastic_params = p["train"]["elastic_net_param"] if isinstance(p["train"]["elastic_net_param"], list) else [p["train"]["elastic_net_param"]]
                
                paramGrid = ParamGridBuilder() \
                    .addGrid(clf.maxIter, max_iters) \
                    .addGrid(clf.regParam, reg_params) \
                    .addGrid(clf.elasticNetParam, elastic_params) \
                    .build()
                    
            elif algo == "random_forest":
                clf = RandomForestClassifier(featuresCol="features", labelCol=label_col)
                
                # Get hyperparameter lists from params.yaml
                rf_trees = p["train"]["rf_trees"] if isinstance(p["train"]["rf_trees"], list) else [p["train"]["rf_trees"]]
                rf_depths = p["train"]["rf_max_depth"] if isinstance(p["train"]["rf_max_depth"], list) else [p["train"]["rf_max_depth"]]
                
                paramGrid = ParamGridBuilder() \
                    .addGrid(clf.numTrees, rf_trees) \
                    .addGrid(clf.maxDepth, rf_depths) \
                    .build()
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

            evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="f1")
            crossval = CrossValidator(
                estimator=clf,
                estimatorParamMaps=paramGrid,
                evaluator=evaluator,
                numFolds=3,
                seed=42
            )

            # Run cross-validation
            cv_model = crossval.fit(train)
            param_maps = cv_model.getEstimatorParamMaps()
            avg_metrics = cv_model.avgMetrics

            print(f"\n[Cross-Validation Results for {algo}]")
            print(f"Total trials: {len(param_maps)}")
            print("-" * 60)

            # Log each parameter combination in nested runs
            for idx, params in enumerate(param_maps):
                param_dict = {k.name: v for k, v in params.items()}
                print(f"Trial {idx+1}: {param_dict}")
                print(f"  → Validation F1: {avg_metrics[idx]:.4f}")
                
                with mlflow.start_run(run_name=f"{algo}_cv_trial_{idx+1}", nested=True):
                    mlflow.log_params(param_dict)
                    mlflow.log_metric("cv_f1", avg_metrics[idx])
                    mlflow.log_param("algorithm", algo)

            # Get best model for this algorithm
            best_model = cv_model.bestModel
            preds = best_model.transform(test)
            f1 = evaluator.evaluate(preds)
            acc = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy").evaluate(preds)

            print("-" * 60)
            print(f"[Algorithm Results] {algo} - Test F1: {f1:.4f}, Test Accuracy: {acc:.4f}")

            # Log algorithm-specific results in nested run
            with mlflow.start_run(run_name=f"{algo}_best_model", nested=True):
                if algo == "logreg":
                    mlflow.log_param("best_maxIter", best_model._java_obj.getMaxIter())
                    mlflow.log_param("best_regParam", best_model._java_obj.getRegParam())
                    mlflow.log_param("best_elasticNetParam", best_model._java_obj.getElasticNetParam())
                else:
                    mlflow.log_param("best_numTrees", best_model._java_obj.getNumTrees())
                    mlflow.log_param("best_maxDepth", best_model._java_obj.getMaxDepth())
                
                mlflow.log_param("algorithm", algo)
                mlflow.log_metric("test_f1", f1)
                mlflow.log_metric("test_accuracy", acc)

            # Track best overall model
            if f1 > best_overall_f1:
                best_overall_f1 = f1
                best_overall_model = best_model
                best_algo = algo

        # Log final best results in parent run
        print("\n" + "=" * 60)
        print(f"[Final Results] Best Algorithm: {best_algo}")
        print(f"Best Test F1: {best_overall_f1:.4f}")
        print("=" * 60)

        if best_algo == "logreg":
            mlflow.log_param("final_best_maxIter", best_overall_model._java_obj.getMaxIter())
            mlflow.log_param("final_best_regParam", best_overall_model._java_obj.getRegParam())
            mlflow.log_param("final_best_elasticNetParam", best_overall_model._java_obj.getElasticNetParam())
        else:
            mlflow.log_param("final_best_numTrees", best_overall_model._java_obj.getNumTrees())
            mlflow.log_param("final_best_maxDepth", best_overall_model._java_obj.getMaxDepth())

        mlflow.log_param("final_best_algorithm", best_algo)
        mlflow.log_metric("final_best_f1", best_overall_f1)
        
        # Re-evaluate best model for accuracy
        final_preds = best_overall_model.transform(test)
        final_acc = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy").evaluate(final_preds)
        mlflow.log_metric("final_best_accuracy", final_acc)

        # Save best model
        os.makedirs("models", exist_ok=True)
        best_overall_model.save("models/spark_model")

    spark.stop()
