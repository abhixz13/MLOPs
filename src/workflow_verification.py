import os
import mlflow
import uuid
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from src.data.loader import load_data
from src.data.preprocessor import DataPreprocessor


class WorkflowVerification:
    def __init__(self):
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor()

        # Generate a unique and meaningful run identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = str(uuid.uuid4())[:8]
        self.run_name = f"Verification_{run_id}"

        mlflow.set_experiment('HousePricePrediction_Verification')

        # Define artifact path
        # Ensure the artifact directory exists
        self.artifact_path = 'mlflow/artifacts/linear_regression'
        os.makedirs(self.artifact_path, exist_ok=True)

    def run_verification(self):
        """
        Comprehensive end-to-end machine learning workflow verification
        """
        print("🔍 Starting End-to-End Machine Learning Workflow Verification\n")

        # 1. Data Loading
        try:
            data = load_data()
            print("✅ Data Loading: Successful")
            print(f"   Data Shape: {data.shape}")
        except Exception as e:
            print(f"❌ Data Loading Failed: {e}")
            return

        # 2. Data Preprocessing
        try:
            X, y = self.preprocessor.preprocess(data, training=True)
            X_train, X_test, y_train, y_test = self.preprocessor.split_data(
                X, y)

            print("✅ Data Preprocessing: Successful")
            print(f"   Training Features Shape: {X_train.shape}")
            print(f"   Test Features Shape: {X_test.shape}")
        except Exception as e:
            print(f"❌ Data Preprocessing Failed: {e}")
            return

        # 3. Model Training and Detailed Evaluation
        with mlflow.start_run(run_name="LinearRegression_Verification"):

            try:
                # Log additional metadata to help track runs
                mlflow.log_param('run_timestamp', datetime.now().isoformat())
                mlflow.log_param('unique_run_id', self.run_name)

                # Train Linear Regression
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)

                # Comprehensive Metrics
                metrics = {
                    'mean_squared_error': mean_squared_error(y_test, y_pred),
                    'r2_score': r2_score(y_test, y_pred),
                    'mean_absolute_error': mean_absolute_error(y_test, y_pred)
                }

                # Log metrics
                mlflow.log_metrics(metrics)

                # Prepare input example for model signature
                input_example = X_train.iloc[:5]

                # Log the model with a specific artifact path and input example
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=self.artifact_path,
                    input_example=input_example,
                    registered_model_name="LinearRegressionModel"
                )

                # Save scaler to the artifact path
                scaler_path = os.path.join(self.artifact_path, 'scaler.joblib')
                self.preprocessor.save_scaler(scaler_path)
                mlflow.log_artifact(scaler_path)

                print("\n✅ Model Training: Successful")
                for metric, value in metrics.items():
                    print(f"   {metric.replace('_', ' ').title()}: {value}")

            except Exception as e:
                # Log any errors explicitly
                mlflow.log_param('run_status', 'failed')
                mlflow.log_param('error_message', str(e))
                raise
                '''
                print(f"❌ Model Training Failed: {e}")
                # Optionally, print full traceback for debugging
                import traceback
                traceback.print_exc()
                return
                '''

        print("\n🎉 Complete End-to-End Workflow Verification Passed Successfully!")


def main():
    verification = WorkflowVerification()
    verification.run_verification()


if __name__ == "__main__":
    main()
