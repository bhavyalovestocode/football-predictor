from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR


RAW_FILE = RAW_DATA_DIR / "ucl_matches_raw.csv"


def load_raw_matches() -> pd.DataFrame:
    """
    Load the raw UEFA Champions League match dataset.

    Returns
    -------
    pd.DataFrame
        Raw UCL match data.
    """

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw UCL dataset not found: {RAW_FILE}"
        )

    df = pd.read_csv(RAW_FILE, sep=";")

    return df


if __name__ == "__main__":
    matches = load_raw_matches()

    print("UCL dataset loaded successfully.")
    print(f"Rows: {len(matches)}")
    print(f"Columns: {len(matches.columns)}")
    print()
    print("Columns:")
    print(matches.columns.tolist())