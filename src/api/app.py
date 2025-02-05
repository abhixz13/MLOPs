# src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict
import mlflow
import logging
from src.models.predict import ModelPredictor
from src.data.preprocessor import DataPreprocessor

# Initialize FastAPI app
app = FastAPI(title="House Price Prediction Service")

# Initialize logger
logger = logging.getLogger(__name__)

# Define request/response models


class PredictionRequest(BaseModel):
    house_size: float
    num_rooms: int
    location_score: float


class BatchPredictionRequest(BaseModel):
    instances: List[PredictionRequest]


class PredictionResponse(BaseModel):
    predicted_price: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str


# Global objects for model and preprocessor
model_predictor = None
preprocessor = None


@app.on_event("startup")
async def load_model():
    """Load the model and preprocessor on startup."""
    global model_predictor, preprocessor
    try:
        # Load the latest model from MLflow
        model_path = "runs:/latest/model"  # You might want to specify exact run_id
        model_predictor = ModelPredictor(model_path)

        # Initialize and load preprocessor
        preprocessor = DataPreprocessor()
        preprocessor.load_scaler("models/scaler.joblib")

        logger.info("Model and preprocessor loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError("Failed to load model and preprocessor")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction."""
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([{
            'house_size': request.house_size,
            'num_rooms': request.num_rooms,
            'location_score': request.location_score
        }])

        # Preprocess input
        X_processed, _ = preprocessor.preprocess(input_data, training=False)

        # Make prediction
        prediction = model_predictor.predict(X_processed)[0]

        return PredictionResponse(
            predicted_price=float(prediction),
            model_version="latest"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Make batch predictions."""
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([{
            'house_size': req.house_size,
            'num_rooms': req.num_rooms,
            'location_score': req.location_score
        } for req in request.instances])

        # Preprocess input
        X_processed, _ = preprocessor.preprocess(input_data, training=False)

        # Make predictions
        predictions = model_predictor.predict(X_processed)

        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            model_version="latest"
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model_predictor is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}
