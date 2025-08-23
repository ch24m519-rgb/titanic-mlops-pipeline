# Titanic MLOps (Spark + DVC + MLflow + FastAPI)

End-to-end pipeline for Titanic classification with Spark preprocessing/training, DVC versioning, MLflow tracking/registry, FastAPI serving, drift detection, and auto-retrain.

## 1. Prereqs

- Python 3.10 (or `conda env create -f environment.yml`)
- Docker + Docker Compose
- (Optional) Kaggle CLI with credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`)

## 2. Setup

```bash
cp .env.example .env
pip install -r requirements.txt


get_data → preprocess → split → train → evaluate → register_best → drift_check → retrain_if_drift