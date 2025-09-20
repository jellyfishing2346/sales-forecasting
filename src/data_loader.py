import pandas as pd

def load_data(filepath):
    """Load sales data from a CSV file."""
    df = pd.read_csv(filepath, parse_dates=True, infer_datetime_format=True)
    return df

# Add more data cleaning functions as needed
