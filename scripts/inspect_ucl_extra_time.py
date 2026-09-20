import pandas as pd

from src.config import RAW_DATA_DIR

RAW_FILE = RAW_DATA_DIR / "ucl_matches_raw.csv"


def main():
    df = pd.read_csv(RAW_FILE, sep=";")

    print("=" * 70)
    print("UCL EXTRA-TIME / PENALTY DATA DIAGNOSTIC")
    print("=" * 70)

    print("\nRaw shape:")
    print(df.shape)

    print("\nRaw missing values:")
    print(
        df[
            [
                "ft1",
                "ft2",
                "et1",
                "et2",
                "pen1",
                "pen2",
            ]
        ].isna().sum()
    )

    print("\nRows with extra-time fields populated:")
    et_rows = df[df["et1"].notna() | df["et2"].notna()]

    print(f"Count: {len(et_rows)}")

    if len(et_rows) > 0:
        print(
            et_rows[
                [
                    "season",
                    "date",
                    "t1_name",
                    "ft1",
                    "ft2",
                    "et1",
                    "et2",
                    "pen1",
                    "pen2",
                    "phase",
                    "round",
                    "match_type",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    print("\nRows with penalty data:")
    penalty_rows = df[df["pen1"].notna() | df["pen2"].notna()]

    print(f"Count: {len(penalty_rows)}")

    if len(penalty_rows) > 0:
        print(
            penalty_rows[
                [
                    "season",
                    "date",
                    "t1_name",
                    "ft1",
                    "ft2",
                    "et1",
                    "et2",
                    "pen1",
                    "pen2",
                    "phase",
                    "round",
                    "match_type",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    print("\nSample ordinary matches:")
    ordinary = df[
        df["et1"].isna()
        & df["et2"].isna()
        & df["pen1"].isna()
        & df["pen2"].isna()
    ]

    print(
        ordinary[
            [
                "season",
                "date",
                "t1_name",
                "ft1",
                "ft2",
                "et1",
                "et2",
                "pen1",
                "pen2",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()