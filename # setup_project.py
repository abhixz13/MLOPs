# setup_project.py
import os
import pathlib


def create_directory_structure():
    """Create the project directory structure with placeholder files."""

    # Define the directory structure
    directories = [
        "src/data",
        "src/models",
        "src/api",
        "src/utils",
        "tests/data",
        "tests/models",
        "tests/api",
        "tests/utils",
        "airflow/dags",
        "airflow/docker",
        "mlflow/docker",
        "notebooks",
        "config",
        "docker"
    ]

    # Create directories
    for directory in directories:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        # Create __init__.py in Python package directories
        if directory.startswith(("src/", "tests/")):
            init_file = pathlib.Path(directory) / "__init__.py"
            init_file.touch()

    # Create essential files with placeholder content
    files = {
        "README.md": """# ML Deployment Project
        
This project implements an end-to-end MLOps pipeline with:
- Model training and evaluation
- MLflow for experiment tracking
- Airflow for orchestration
- Docker for containerization
- FastAPI for model serving
""",

        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDEs
.idea/
.vscode/
*.swp
*.swo

# MLflow
mlruns/

# Airflow
airflow/logs/
airflow/*.pid
airflow/*.err
airflow/*.out

# Jupyter
.ipynb_checkpoints
""",

        "requirements.txt": """# Core ML and Data Processing
numpy>=1.24.3
pandas>=2.0.2
scikit-learn>=1.2.2

# MLflow for experiment tracking
mlflow>=2.3.1

# FastAPI for API development
fastapi>=0.95.2
uvicorn>=0.22.0
python-multipart>=0.0.6

# Airflow and its dependencies
apache-airflow>=2.6.1

# Monitoring and logging
prometheus-client>=0.17.0
python-json-logger>=2.0.7

# Testing
pytest>=7.3.1

# Development tools
black>=23.3.0
flake8>=6.0.0
python-dotenv>=1.0.0
""",

        "Dockerfile": """FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
""",

        "docker-compose.yml": """version: '3'

services:
  mlflow:
    build: 
      context: .
      dockerfile: mlflow/docker/Dockerfile
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlruns
    environment:
      - MLFLOW_TRACKING_URI=http://localhost:5000

  airflow:
    build:
      context: .
      dockerfile: airflow/docker/Dockerfile
    ports:
      - "8080:8080"
    volumes:
      - ./airflow/dags:/opt/airflow/dags
    environment:
      - AIRFLOW__CORE__LOAD_EXAMPLES=False
      - MLFLOW_TRACKING_URI=http://mlflow:5000

  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - mlflow
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000

volumes:
  mlflow-data:
"""
    }

    # Create files
    for file_path, content in files.items():
        with open(file_path, 'w') as f:
            f.write(content.lstrip())

    print("Project structure created successfully!")


if __name__ == "__main__":
    create_directory_structure()
