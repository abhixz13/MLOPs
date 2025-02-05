# airflow/dags/training_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow
import pandas as pd
import os
from src.data.loader import generate_sample_data
from src.data.preprocessor import DataPreprocessor
from src.models.train import ModelTrainer

import sys
from pathlib import Path

# Add src directory to Python path
dag_dir = Path(__file__).parent.parent
src_dir = dag_dir.parent / 'src'
sys.path.append(str(src_dir))

# Define default arguments
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Create DAG
dag = DAG(
    'house_price_prediction_training',
    default_args=default_args,
    description='Training pipeline for house price prediction model',
    schedule_interval='0 0 * * 0',  # Run weekly at midnight on Sunday
    catchup=False
)


def generate_and_preprocess_data(**context):
    """Generate sample data and preprocess it."""
    try:
        # Generate data
        data = generate_sample_data(n_samples=10000)

        # Initialize preprocessor
        preprocessor = DataPreprocessor()

        # Preprocess data
        X, y = preprocessor.preprocess(data, training=True)

        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

        # Save preprocessor
        preprocessor.save_scaler('models/scaler.joblib')

        # Push to XCom
        context['task_instance'].xcom_push(key='training_data', value={
            'X_train': X_train.to_dict(),
            'y_train': y_train.to_list(),
            'X_test': X_test.to_dict(),
            'y_test': y_test.to_list()
        })

        return "Data processing completed successfully"
    except Exception as e:
        raise Exception(f"Data processing failed: {str(e)}")


def train_model(**context):
    """Train the model and log to MLflow."""
    try:
        # Get data from XCom
        data = context['task_instance'].xcom_pull(
            task_ids='generate_and_preprocess_data',
            key='training_data'
        )

        # Convert back to proper format
        X_train = pd.DataFrame(data['X_train'])
        y_train = pd.Series(data['y_train'])
        X_test = pd.DataFrame(data['X_test'])
        y_test = pd.Series(data['y_test'])

        # Initialize and train model
        trainer = ModelTrainer()
        trainer.train_with_mlflow(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            experiment_name="house_price_prediction"
        )

        return "Model training completed successfully"
    except Exception as e:
        raise Exception(f"Model training failed: {str(e)}")


# Define tasks
generate_data_task = PythonOperator(
    task_id='generate_and_preprocess_data',
    python_callable=generate_and_preprocess_data,
    dag=dag
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

# Set task dependencies
generate_data_task >> train_model_task
