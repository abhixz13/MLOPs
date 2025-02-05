import os


def get_mlflow_config():
    """
    Get MLflow configuration

    Returns:
        dict: MLflow configuration parameters
    """
    return {
        'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
        'experiment_name': os.getenv('MLFLOW_EXPERIMENT_NAME', 'LogisticRegressionExperiments'),
        'artifacts_dir': os.getenv('MLFLOW_ARTIFACTS_DIR', './mlruns')
    }


def get_deployment_config():
    """
    Get model deployment configuration

    Returns:
        dict: Deployment configuration parameters
    """
    return {
        'production_threshold': {
            'accuracy': 0.85,
            'f1_score': 0.80
        },
        'deployment_environment': os.getenv('DEPLOYMENT_ENV', 'staging'),
        'model_version': os.getenv('MODEL_VERSION', '1.0.0')
    }
