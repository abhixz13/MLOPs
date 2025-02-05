import pandas as pd
import numpy as np
from typing import Tuple, Union


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic real estate data for regression analysis.

    This function creates a simulated dataset of house prices based on:
    - House size (in sq ft)
    - Number of rooms
    - Location score

    Args:
        n_samples (int): Number of data points to generate. Defaults to 1000.

    Returns:
        pd.DataFrame: Synthetic real estate dataset with features and target variable
    """
    # Set a fixed random seed for reproducibility
    np.random.seed(42)

    # Feature generation with realistic variations
    # Mean 1500 sq ft, std dev 500
    house_size = np.random.normal(1500, 500, n_samples)
    num_rooms = np.random.randint(1, 8, n_samples)      # 1 to 7 rooms
    location_score = np.random.uniform(
        1, 10, n_samples)  # Location quality score 1-10

    # Price generation with realistic correlation to features
    house_price = (
        200 * house_size +      # Price per sq ft
        50000 * num_rooms +      # Premium for additional rooms
        25000 * location_score +  # Location value
        np.random.normal(0, 50000, n_samples)  # Random noise
    )

    # Create DataFrame with descriptive column names
    data = pd.DataFrame({
        'house_size': house_size,
        'num_rooms': num_rooms,
        'location_score': location_score,
        'house_price': house_price
    })

    return data


def load_data() -> pd.DataFrame:
    """
    Primary data loading function for the project.

    Returns:
        pd.DataFrame: Generated or loaded dataset for house price prediction
    """
    # In future, this could be modified to:
    # 1. Load from a CSV file
    # 2. Connect to a database
    # 3. Fetch from an API
    return generate_sample_data()


def main():
    """
    Quick data generation and validation script.
    Useful for testing and understanding the data generation process.
    """
    data = generate_sample_data()

    print("🏠 Synthetic Real Estate Dataset Overview")
    print(f"Total Samples: {len(data)}")
    print("\nDescriptive Statistics:")
    print(data.describe())

    # Optional: Basic visualizations could be added here
    # import matplotlib.pyplot as plt
    # data.hist(figsize=(10, 6))
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
