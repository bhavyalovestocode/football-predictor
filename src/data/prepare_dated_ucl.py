import pandas as pd

from src.config import PROCESSED_DATA_DIR


INPUT_FILE = PROCESSED_DATA_DIR / "ucl_matches_clean.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "ucl_matches_dated.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        utc=True,
    )

    before = len(df)

    # Temporal feature engineering requires
    # a reliable chronological order.
    df = df.dropna(subset=["date"]).copy()

    df = df.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    after = len(df)

    print("=" * 70)
    print("PREPARING DATED UCL DATASET")
    print("=" * 70)

    print(f"\nInput matches: {before}")
    print(f"Matches with valid dates: {after}")
    print(f"Excluded matches: {before - after}")

    print("\nDate range:")
    print(df["date"].min())
    print(df["date"].max())

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()