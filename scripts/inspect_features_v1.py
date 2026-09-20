import pandas as pd

from src.config import PROCESSED_DATA_DIR


FILE = PROCESSED_DATA_DIR / "ucl_features_v1.csv"


def main():
    df = pd.read_csv(FILE)

    print("=" * 70)
    print("TEMPORAL UCL FEATURES V1")
    print("=" * 70)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nTarget distribution:")
    print(df["target"].value_counts())

    print("\nTarget percentages:")
    print(
        df["target"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nMissing values:")
    print(
        df.isna().sum()
        .sort_values(ascending=False)
        .head(20)
    )

    print("\nFirst 10 feature rows:")
    print(
        df[
            [
                "date",
                "home_team",
                "away_team",
                "home_form_matches",
                "home_form_ppg",
                "away_form_matches",
                "away_form_ppg",
                "home_overall_ppg",
                "away_overall_ppg",
                "home_team_home_ppg",
                "away_team_away_ppg",
                "target",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nLast 10 feature rows:")
    print(
        df[
            [
                "date",
                "home_team",
                "away_team",
                "home_form_ppg",
                "away_form_ppg",
                "target",
            ]
        ]
        .tail(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()