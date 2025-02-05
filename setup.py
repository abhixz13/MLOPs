# setup.py
from setuptools import setup, find_packages  # type: ignore

setup(
    name="Project_mediassist",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.2",
        "scikit-learn>=1.2.2",
        "mlflow>=2.3.1",
        "fastapi>=0.95.2",
        "uvicorn>=0.22.0",
    ],
)
