from src.data.load_ucl import load_raw_matches


def main():
    df = load_raw_matches()

    print("=" * 70)
    print("UEFA CHAMPIONS LEAGUE DATASET INSPECTION")
    print("=" * 70)

    print("\nDataset shape:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nMatch status:")
    print(df["status"].value_counts(dropna=False))

    print("\nSeasons:")
    print(df["season"].nunique())

    print("\nEarliest season:")
    print(df["season"].min())

    print("\nLatest season:")
    print(df["season"].max())

    print("\nUnique teams:")
    teams = set(df["t1_name"].dropna()) | set(df["t2_name"].dropna())
    print(len(teams))

    print("\nSample teams:")
    for team in sorted(teams)[:20]:
        print(f"  {team}")


if __name__ == "__main__":
    main()