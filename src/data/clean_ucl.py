import pandas as pd

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


RAW_FILE = RAW_DATA_DIR / "ucl_matches_raw.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "ucl_matches_clean.csv"


def load_data() -> pd.DataFrame:
    return pd.read_csv(RAW_FILE, sep=";")


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Keep only completed matches with valid 90-minute scores.
    df = df[df["status"] == "FINISHED"].copy()
    df = df.dropna(subset=["ft1", "ft2"]).copy()

    # Parse the source timestamps correctly.
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        utc=True,
    )

    df["ft1"] = df["ft1"].astype(int)
    df["ft2"] = df["ft2"].astype(int)

    # Result after 90 minutes.
    df["result_after_90"] = "DRAW"

    df.loc[
        df["ft1"] > df["ft2"],
        "result_after_90"
    ] = "HOME_WIN"

    df.loc[
        df["ft1"] < df["ft2"],
        "result_after_90"
    ] = "AWAY_WIN"

    # et1 / et2 represent the score after extra time.
    # In ordinary matches they are identical to ft1 / ft2.
    df["extra_time"] = (
        (df["et1"] != df["ft1"])
        | (df["et2"] != df["ft2"])
    )

    # Result after extra time.
    df["result_after_extra_time"] = "DRAW"

    df.loc[
        df["et1"] > df["et2"],
        "result_after_extra_time"
    ] = "HOME_WIN"

    df.loc[
        df["et1"] < df["et2"],
        "result_after_extra_time"
    ] = "AWAY_WIN"

    # Penalty shootout.
    df["went_to_penalties"] = (
        df["pen1"].notna()
        & df["pen2"].notna()
    )

    clean = df[
        [
            "match_id",
            "season",
            "date",
            "t1",
            "t1_name",
            "t2",
            "t2_name",
            "ft1",
            "ft2",
            "et1",
            "et2",
            "pen1",
            "pen2",
            "result_after_90",
            "result_after_extra_time",
            "extra_time",
            "went_to_penalties",
            "phase",
            "round",
            "leg",
            "group",
            "format",
            "match_type",
            "mode",
            "mode_detail",
        ]
    ].copy()

    clean = clean.rename(
        columns={
            "t1": "home_team_id",
            "t1_name": "home_team",
            "t2": "away_team_id",
            "t2_name": "away_team",
            "ft1": "home_goals",
            "ft2": "away_goals",
        }
    )

    clean = clean.sort_values(
        ["date", "match_id"],
        na_position="last"
    ).reset_index(drop=True)

    return clean


def main():
    print("Loading raw UCL data...")

    df = load_data()

    print(f"Raw matches: {len(df)}")

    clean = clean_matches(df)

    print(f"Clean matches: {len(clean)}")

    clean.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Saved cleaned dataset to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()