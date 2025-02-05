import mlflow
import mlflow.sklearn
import numpy as np
from utils.config import get_deployment_config
from utils.logger import setup_logger


class ModelDeployer:
    def __init__(self, model_name):
        """
        Initialize Model Deployer

        Args:
            model_name (str): Name of the model to deploy
        """
        self.logger = setup_logger(__name__)
        self.model_name = model_name
        self.config = get_deployment_config()

    def load_best_model(self):
        """
        Load the best model from MLflow registry

        Returns:
            tuple: Loaded model and preprocessor
        """
        try:
            # Load model from MLflow registry
            model = mlflow.sklearn.load_model(
                f"models:/{self.model_name}/latest")

            # Load preprocessor (scaler)
            scaler = mlflow.sklearn.load_model(f"models:/FeatureScaler/latest")

            self.logger.info(f"Successfully loaded model {self.model_name}")
            return model, scaler
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def predict(self, X):
        """
        Make predictions using the deployed model

        Args:
            X (numpy.ndarray): Input features

        Returns:
            numpy.ndarray: Predicted labels
        """
        try:
            # Load model and scaler
            model, scaler = self.load_best_model()

            # Preprocess input
            X_scaled = scaler.transform(X)

            # Make predictions
            predictions = model.predict(X_scaled)

            return predictions
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            raise

    def deploy_model(self):
        """
        Deploy the model to the specified environment
        """
        try:
            # Retrieve the latest model version
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(
                self.model_name, stages=['None'])[0]

            # Transition model to production stage
            client.transition_model_version_stage(
                name=self.model_name,
                version=model_version.version,
                stage="Production"
            )

            self.logger.info(f"Model {self.model_name} deployed to production")
        except Exception as e:
            self.logger.error(f"Model deployment error: {e}")
            raise


def main():
    # Deploy the logistic regression model
    deployer = ModelDeployer("LogisticRegressionModel")
    deployer.deploy_model()

    # Optionally, you can add a test prediction
    # from data.loader import load_test_data
    # X_test = load_test_data()
    # predictions = deployer.predict(X_test)
    # print("Test Predictions:", predictions)


if __name__ == "__main__":
    main()
