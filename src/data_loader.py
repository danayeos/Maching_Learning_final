import pandas as pd

def load_data(path: str = "data/fullcifocoss.csv", sep: str = ";") -> pd.DataFrame:
    """Load the main food consumption dataset."""
    df = pd.read_csv(path, sep=sep)
    return df
