from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import joblib
import os


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = ['house_size', 'num_rooms', 'location_score']
        self.target_column = 'house_price'

    def preprocess(self, data: pd.DataFrame, training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the data for training or prediction.
        """
        # Separate features and target
        X = data[self.feature_columns]
        y = data[self.target_column] if self.target_column in data.columns else None

        # Scale features
        if training:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=self.feature_columns
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=self.feature_columns
            )

        return X_scaled, y

    def save_scaler(self, path: str):
        """Save the fitted scaler."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)

    def load_scaler(self, path: str):
        """Load a fitted scaler."""
        self.scaler = joblib.load(path)

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=42)
