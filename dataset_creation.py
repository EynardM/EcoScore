import pandas as pd
from datasets import Dataset

def create_dataset_from_csv(csv_path):
    """
    Create a Dataset object from a CSV file.

    Parameters:
    - csv_path (str): Path to the CSV file.

    Returns:
    - datasets.Dataset: Dataset object containing the data from the CSV file.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Convert DataFrame to a dictionary
    dataset_dict = {
        "review": df['review'].tolist(),
        "category": df['category'].tolist(),
        "topic": df['topic'].tolist()
    }

    # Create a Dataset object
    dataset = Dataset.from_dict(dataset_dict)

    return dataset

# Example usage:
csv_path = 'reviews_dataset.csv'
reviews_dataset = create_dataset_from_csv(csv_path)
print(reviews_dataset)