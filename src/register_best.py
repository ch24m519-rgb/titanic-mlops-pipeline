import os, json, mlflow
from src.utils import read_params

if __name__ == "__main__":
    p = read_params()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000"))
    mlflow.set_experiment(p["mlflow"]["experiment_name"])
    model_name = p["mlflow"]["model_name"]
    metric = p["mlflow"]["metric_to_maximize"]

    # Find best run in this experiment by metric
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(p["mlflow"]["experiment_name"])
    runs = client.search_runs([exp.experiment_id], order_by=[f"metrics.{metric} DESC"], max_results=1)
    if not runs:
        raise SystemExit("No runs found to register")

    best = runs[0]
    run_id = best.info.run_id
    uri = f"runs:/{run_id}/model"

    # Register new version
    mv = mlflow.register_model(uri, model_name)

    # Transition last version to Production (simple policy)
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )

