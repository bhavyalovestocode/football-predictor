import pandas as pd

from src.config import PROCESSED_DATA_DIR


FILE = PROCESSED_DATA_DIR / "ucl_matches_clean.csv"


def main():
    df = pd.read_csv(FILE)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        utc=True,
    )

    print("=" * 70)
    print("CLEAN UCL DATASET")
    print("=" * 70)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nResult after 90 minutes:")
    print(df["result_after_90"].value_counts())

    print("\nResult after extra time:")
    print(df["result_after_extra_time"].value_counts())

    print("\nExtra-time matches:")
    print(df["extra_time"].value_counts())

    print("\nPenalty shootouts:")
    print(df["went_to_penalties"].value_counts())

    print("\nDate range:")
    print(df["date"].min())
    print(df["date"].max())

    print("\nMatches with missing dates:")
    print(df["date"].isna().sum())

    print("\nFirst 10 matches:")
    print(
        df[
            [
                "date",
                "season",
                "home_team",
                "home_goals",
                "away_goals",
                "away_team",
                "result_after_90",
                "result_after_extra_time",
                "extra_time",
                "went_to_penalties",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()