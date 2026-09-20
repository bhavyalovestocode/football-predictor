import argparse
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR


INPUT_FILE = PROCESSED_DATA_DIR / "ucl_features_v2.csv"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VAL_FILE = PROCESSED_DATA_DIR / "val.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"


def split_dataset(input_file=INPUT_FILE, output_dir=PROCESSED_DATA_DIR):
    df = pd.read_csv(input_file)

    if "date" not in df.columns:
        raise ValueError("Input dataset must contain a 'date' column.")

    if "target" not in df.columns:
        raise ValueError("Input dataset must contain a 'target' column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    if df["date"].isna().any():
        invalid_count = int(df["date"].isna().sum())
        raise ValueError(
            f"Input dataset contains {invalid_count} invalid or missing dates."
        )

    sort_columns = ["date"]
    if "match_id" in df.columns:
        sort_columns.append("match_id")
    df = df.sort_values(sort_columns).reset_index(drop=True)

    total_matches = len(df)
    train_end = timestamp_boundary(df, int(total_matches * 0.70))
    val_end = timestamp_boundary(df, int(total_matches * 0.85))

    splits = {
        "Train": df.iloc[:train_end].copy(),
        "Validation": df.iloc[train_end:val_end].copy(),
        "Test": df.iloc[val_end:].copy(),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        "Train": output_dir / "train.csv",
        "Validation": output_dir / "val.csv",
        "Test": output_dir / "test.csv",
    }

    for split_name, split_df in splits.items():
        split_df.to_csv(output_files[split_name], index=False)

    print_split_summaries(splits)
    print("\nSaved:")
    for output_file in output_files.values():
        print(output_file)

    return splits


def timestamp_boundary(df, requested_boundary):
    boundary = requested_boundary

    while (
        boundary < len(df)
        and boundary > 0
        and df.iloc[boundary]["date"] == df.iloc[boundary - 1]["date"]
    ):
        boundary += 1

    return boundary


def print_split_summaries(splits):
    print("=" * 70)
    print("CHRONOLOGICAL UCL DATASET SPLITS")
    print("=" * 70)

    for split_name, split_df in splits.items():
        print(f"\n{split_name}:")
        print(f"Start date: {split_df['date'].min()}")
        print(f"End date: {split_df['date'].max()}")
        print(f"Total matches: {len(split_df)}")
        print("Target class counts:")
        for target, count in split_df["target"].value_counts().sort_index().items():
            print(f"  {target}: {count}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split temporal UCL features into chronological datasets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help="Input feature CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory for train.csv, val.csv, and test.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_dataset(input_file=args.input, output_dir=args.output_dir)