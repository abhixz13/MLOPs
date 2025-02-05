import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class ModelTrainer:
    def __init__(self, experiment_name='LogisticRegressionExperiment'):
        """
        Initialize MLflow experiment tracking

        Args:
            experiment_name (str): Name of the MLflow experiment
        """
        mlflow.set_experiment(experiment_name)

    def preprocess_data(self, X, y):
        """
        Preprocess data by splitting and scaling

        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector

        Returns:
            tuple: Processed train and test sets
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    def train_with_grid_search(self, X, y):
        """
        Perform grid search for hyperparameter tuning

        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector

        Returns:
            dict: Best model details and performance metrics
        """
        # Preprocess data
        X_train, X_test, y_train, y_test, scaler = self.preprocess_data(X, y)

        # Hyperparameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'lbfgs']
        }

        # Initialize logistic regression
        lr = LogisticRegression(max_iter=1000)

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=lr,
            param_grid=param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )

        # Start MLflow run
        with mlflow.start_run():
            # Fit grid search
            grid_search.fit(X_train, y_train)

            # Get best model
            best_model = grid_search.best_estimator_

            # Predictions
            y_pred = best_model.predict(X_test)

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }

            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Log parameters
            mlflow.log_params(grid_search.best_params_)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="logistic_regression_model",
                registered_model_name="LogisticRegressionModel"
            )

            # Log scaler
            mlflow.sklearn.log_model(
                sk_model=scaler,
                artifact_path="feature_scaler",
                registered_model_name="FeatureScaler"
            )

            return {
                'best_model': best_model,
                'best_params': grid_search.best_params_,
                'metrics': metrics,
                'confusion_matrix': conf_matrix,
                'scaler': scaler
            }


def main():
    """
    Main training script
    """
    # Import data loading and preprocessing functions
    from src.data.loader import load_data
    from src.data.preprocessor import preprocess_data

    # Load data
    X, y = load_data()

    # Preprocess data (if needed)
    X = preprocess_data(X)

    # Initialize trainer
    trainer = ModelTrainer()

    # Run grid search and train model
    result = trainer.train_with_grid_search(X, y)

    # Print model performance
    print("Best Model Metrics:")
    for metric, value in result['metrics'].items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
