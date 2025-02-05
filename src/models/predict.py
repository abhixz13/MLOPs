import logging
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelPredictor:
    def __init__(self, model_path: str = None):
        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load the trained model."""
        self.model = mlflow.sklearn.load_model(model_path)
        logger.info("Model loaded successfully")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the loaded model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")

        predictions = self.model.predict(X)
        return predictions
