import os
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd
from schema import Passenger

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "titanic_clf")

app = FastAPI(title="Titanic Inference API")

@app.on_event("startup")
def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Load the current Production model from registry
    global model
    model_uri = f"models:/{MODEL_NAME}/Production"
    model = mlflow.pyfunc.load_model(model_uri)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(passenger: Passenger):
    df = pd.DataFrame([passenger.model_dump()])
    preds = model.predict(df)
    return {"prediction": int(preds[0])}
 
