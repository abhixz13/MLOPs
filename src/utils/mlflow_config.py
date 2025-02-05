import os
import mlflow
from mlflow.tracking import MlflowClient


def configure_mlflow():
    """
    Configure MLflow tracking server settings

    Explanation:
    - Set the tracking URI to point to your MLflow server
    - Ensure experiment tracking is properly initialized
    """
    # Default to local tracking server if not set
    mlflow_tracking_uri = os.getenv(
        'MLFLOW_TRACKING_URI',
        'http://localhost:5001'  # Default local MLflow server
    )

    try:
        # Set the tracking URI
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        print(f"🔗 MLflow Tracking URI set to: {mlflow_tracking_uri}")

        # Create MLflow client
        client = MlflowClient()

        # List experiments (using the correct method)
        experiments = client.search_experiments()

        print(f"✅ Connected to MLflow. Total Experiments: {len(experiments)}")

        return client
    except Exception as e:
        print(f"❌ MLflow Configuration Error: {e}")
        print("Troubleshooting Tips:")
        print("1. Ensure MLflow server is running")
        print("2. Check MLFLOW_TRACKING_URI environment variable")
        print("3. Verify network connectivity")
        return None
