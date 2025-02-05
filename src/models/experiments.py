import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MLflowExperimentTracker:
    def __init__(self, experiment_name):
        """
        Initialize MLflow experiment tracker

        Args:
            experiment_name (str): Name of the MLflow experiment
        """
        # Set the experiment
        mlflow.set_experiment(experiment_name)

    def run_experiment(self, X, y, params=None):
        """
        Run an MLflow experiment for logistic regression

        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector
            params (dict, optional): Hyperparameters for logistic regression

        Returns:
            dict: Experiment metrics and model information
        """
        # Default hyperparameters if not provided
        if params is None:
            params = {
                'C': 1.0,  # Inverse of regularization strength
                'penalty': 'l2',
                'solver': 'lbfgs',
                'max_iter': 1000
            }

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)

            # Train the model
            model = LogisticRegression(**params)
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log the model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="logistic_regression_model",
                registered_model_name="LogisticRegressionModel"
            )

            # Log the scaler for preprocessing
            mlflow.sklearn.log_model(
                sk_model=scaler,
                artifact_path="feature_scaler",
                registered_model_name="FeatureScaler"
            )

            return {
                'model': model,
                'scaler': scaler,
                'metrics': metrics,
                'params': params
            }

    def hyperparameter_tuning(self, X, y):
        """
        Perform hyperparameter tuning using grid search

        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector

        Returns:
            dict: Best model and its performance
        """
        # Define hyperparameter grid
        param_grid = [
            {'C': [0.001, 0.01, 0.1, 1, 10, 100],
             'penalty': ['l1', 'l2'],
             'solver': ['liblinear', 'lbfgs']},
        ]

        best_experiment = None
        best_score = 0

        # Run experiments for each parameter combination
        for params in param_grid:
            experiment_result = self.run_experiment(X, y, params)

            # Select best model based on f1 score
            if experiment_result['metrics']['f1_score'] > best_score:
                best_score = experiment_result['metrics']['f1_score']
                best_experiment = experiment_result

        return best_experiment


def main():
    # Example usage (replace with your actual data loading)
    from data.loader import load_data
    from data.preprocessor import preprocess_data

    # Load and preprocess data
    X, y = load_data()
    X = preprocess_data(X)

    # Initialize experiment tracker
    tracker = MLflowExperimentTracker(
        experiment_name="LogisticRegressionExperiments")

    # Run hyperparameter tuning
    best_model = tracker.hyperparameter_tuning(X, y)

    print("Best Model Metrics:", best_model['metrics'])


if __name__ == "__main__":
    main()
