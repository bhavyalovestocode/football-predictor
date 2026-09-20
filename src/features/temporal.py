import pandas as pd

from src.config import PROCESSED_DATA_DIR


INPUT_FILE = PROCESSED_DATA_DIR / "ucl_matches_dated.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "ucl_features_v1.csv"

FORM_WINDOW = 5


def get_empty_history():
    return {
        "matches": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0,
        "goals_for": 0,
        "goals_against": 0,
        "home_matches": 0,
        "home_wins": 0,
        "home_draws": 0,
        "home_losses": 0,
        "away_matches": 0,
        "away_wins": 0,
        "away_draws": 0,
        "away_losses": 0,
    }


def calculate_form(history):
    matches = history["matches"]

    if matches == 0:
        return {
            "ppg": 0.0,
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
            "goals_for_pg": 0.0,
            "goals_against_pg": 0.0,
        }

    return {
        "ppg": history["points"] / matches,
        "win_rate": history["wins"] / matches,
        "draw_rate": history["draws"] / matches,
        "loss_rate": history["losses"] / matches,
        "goals_for_pg": history["goals_for"] / matches,
        "goals_against_pg": history["goals_against"] / matches,
    }


def calculate_home_form(history):
    matches = history["home_matches"]

    if matches == 0:
        return {
            "ppg": 0.0,
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
        }

    points = (
        history["home_wins"] * 3
        + history["home_draws"]
    )

    return {
        "ppg": points / matches,
        "win_rate": history["home_wins"] / matches,
        "draw_rate": history["home_draws"] / matches,
        "loss_rate": history["home_losses"] / matches,
    }


def calculate_away_form(history):
    matches = history["away_matches"]

    if matches == 0:
        return {
            "ppg": 0.0,
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
        }

    points = (
        history["away_wins"] * 3
        + history["away_draws"]
    )

    return {
        "ppg": points / matches,
        "win_rate": history["away_wins"] / matches,
        "draw_rate": history["away_draws"] / matches,
        "loss_rate": history["away_losses"] / matches,
    }


def update_history(history, goals_for, goals_against, is_home):
    history["matches"] += 1
    history["goals_for"] += goals_for
    history["goals_against"] += goals_against

    if goals_for > goals_against:
        history["wins"] += 1
        history["points"] += 3
    elif goals_for == goals_against:
        history["draws"] += 1
        history["points"] += 1
    else:
        history["losses"] += 1

    if is_home:
        history["home_matches"] += 1

        if goals_for > goals_against:
            history["home_wins"] += 1
        elif goals_for == goals_against:
            history["home_draws"] += 1
        else:
            history["home_losses"] += 1

    else:
        history["away_matches"] += 1

        if goals_for > goals_against:
            history["away_wins"] += 1
        elif goals_for == goals_against:
            history["away_draws"] += 1
        else:
            history["away_losses"] += 1


def add_to_recent_form(recent_matches, goals_for, goals_against):
    recent_matches.append(
        {
            "goals_for": goals_for,
            "goals_against": goals_against,
            "result": (
                "W"
                if goals_for > goals_against
                else "D"
                if goals_for == goals_against
                else "L"
            ),
        }
    )

    if len(recent_matches) > FORM_WINDOW:
        recent_matches.pop(0)


def calculate_recent_form(recent_matches):
    matches = len(recent_matches)

    if matches == 0:
        return {
            "matches": 0,
            "points": 0,
            "ppg": 0.0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goals_for_pg": 0.0,
            "goals_against_pg": 0.0,
        }

    wins = sum(
        match["result"] == "W"
        for match in recent_matches
    )

    draws = sum(
        match["result"] == "D"
        for match in recent_matches
    )

    losses = matches - wins - draws

    goals_for = sum(
        match["goals_for"]
        for match in recent_matches
    )

    goals_against = sum(
        match["goals_against"]
        for match in recent_matches
    )

    points = wins * 3 + draws

    return {
        "matches": matches,
        "points": points,
        "ppg": points / matches,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goals_for_pg": goals_for / matches,
        "goals_against_pg": goals_against / matches,
    }


def build_features(df):
    histories = {}
    recent_forms = {}

    feature_rows = []

    # Process matches in chronological timestamp groups.
    # Every match at the same timestamp sees the exact same
    # team history from before that timestamp.
    for _, timestamp_group in df.groupby("date", sort=False):
        # Calculate features for every match before updating histories.
        for _, match in timestamp_group.iterrows():
            home_id = match["home_team_id"]
            away_id = match["away_team_id"]

            if home_id not in histories:
                histories[home_id] = get_empty_history()
            if away_id not in histories:
                histories[away_id] = get_empty_history()
            if home_id not in recent_forms:
                recent_forms[home_id] = []
            if away_id not in recent_forms:
                recent_forms[away_id] = []

            home_history = histories[home_id]
            away_history = histories[away_id]
            home_recent = calculate_recent_form(recent_forms[home_id])
            away_recent = calculate_recent_form(recent_forms[away_id])
            home_overall = calculate_form(home_history)
            away_overall = calculate_form(away_history)
            home_home = calculate_home_form(home_history)
            away_away = calculate_away_form(away_history)

            row = {
                "match_id": match["match_id"], "date": match["date"],
                "season": match["season"], "home_team_id": home_id,
                "home_team": match["home_team"], "away_team_id": away_id,
                "away_team": match["away_team"],
                "home_form_matches": home_recent["matches"], "home_form_points": home_recent["points"],
                "home_form_ppg": home_recent["ppg"], "home_form_wins": home_recent["wins"],
                "home_form_draws": home_recent["draws"], "home_form_losses": home_recent["losses"],
                "home_form_goals_for": home_recent["goals_for"], "home_form_goals_against": home_recent["goals_against"],
                "home_form_goals_for_pg": home_recent["goals_for_pg"], "home_form_goals_against_pg": home_recent["goals_against_pg"],
                "away_form_matches": away_recent["matches"], "away_form_points": away_recent["points"],
                "away_form_ppg": away_recent["ppg"], "away_form_wins": away_recent["wins"],
                "away_form_draws": away_recent["draws"], "away_form_losses": away_recent["losses"],
                "away_form_goals_for": away_recent["goals_for"], "away_form_goals_against": away_recent["goals_against"],
                "away_form_goals_for_pg": away_recent["goals_for_pg"], "away_form_goals_against_pg": away_recent["goals_against_pg"],
                "home_matches_played": home_history["matches"], "away_matches_played": away_history["matches"],
                "home_overall_ppg": home_overall["ppg"], "away_overall_ppg": away_overall["ppg"],
                "home_overall_win_rate": home_overall["win_rate"], "away_overall_win_rate": away_overall["win_rate"],
                "home_overall_goals_for_pg": home_overall["goals_for_pg"], "away_overall_goals_for_pg": away_overall["goals_for_pg"],
                "home_overall_goals_against_pg": home_overall["goals_against_pg"], "away_overall_goals_against_pg": away_overall["goals_against_pg"],
                "home_team_home_matches": home_history["home_matches"], "away_team_away_matches": away_history["away_matches"],
                "home_team_home_ppg": home_home["ppg"], "away_team_away_ppg": away_away["ppg"],
                "home_team_home_win_rate": home_home["win_rate"], "away_team_away_win_rate": away_away["win_rate"],
                "phase": match["phase"], "round": match["round"], "match_type": match["match_type"],
                "target": match["result_after_90"],
            }
            feature_rows.append(row)

        # Update histories using all completed matches at this timestamp.
        for _, match in timestamp_group.iterrows():
            home_id = match["home_team_id"]
            away_id = match["away_team_id"]
            home_goals = int(match["home_goals"])
            away_goals = int(match["away_goals"])
            update_history(histories[home_id], home_goals, away_goals, is_home=True)
            update_history(histories[away_id], away_goals, home_goals, is_home=False)
            add_to_recent_form(recent_forms[home_id], home_goals, away_goals)
            add_to_recent_form(recent_forms[away_id], away_goals, home_goals)

    return pd.DataFrame(feature_rows)


def main():
    print("=" * 70)
    print("BUILDING TEMPORAL UCL FEATURES")
    print("=" * 70)

    print("\nLoading dated UCL matches...")

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        utc=True,
    )

    df = df.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    print(f"Input matches: {len(df)}")

    features = build_features(df)

    print(f"Feature rows: {len(features)}")
    print(f"Feature columns: {len(features.columns)}")

    features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()