import pandas as pd

from src.features.temporal import build_features


def synthetic_matches():
    return pd.DataFrame(
        [
            {
                "match_id": "before-t",
                "date": pd.Timestamp("2024-01-01T18:00:00Z"),
                "season": "2023/24",
                "home_team_id": 1,
                "home_team": "Team 1",
                "away_team_id": 4,
                "away_team": "Team 4",
                "home_goals": 2,
                "away_goals": 0,
                "phase": "Group",
                "round": "1",
                "match_type": "Competitive",
                "result_after_90": "HOME_WIN",
            },
            {
                "match_id": "game-a",
                "date": pd.Timestamp("2024-01-08T18:00:00Z"),
                "season": "2023/24",
                "home_team_id": 1,
                "home_team": "Team 1",
                "away_team_id": 2,
                "away_team": "Team 2",
                "home_goals": 0,
                "away_goals": 1,
                "phase": "Group",
                "round": "2",
                "match_type": "Competitive",
                "result_after_90": "AWAY_WIN",
            },
            {
                "match_id": "game-b",
                "date": pd.Timestamp("2024-01-08T18:00:00Z"),
                "season": "2023/24",
                "home_team_id": 1,
                "home_team": "Team 1",
                "away_team_id": 3,
                "away_team": "Team 3",
                "home_goals": 3,
                "away_goals": 0,
                "phase": "Group",
                "round": "2",
                "match_type": "Competitive",
                "result_after_90": "HOME_WIN",
            },
            {
                "match_id": "after-t",
                "date": pd.Timestamp("2024-01-09T18:00:00Z"),
                "season": "2023/24",
                "home_team_id": 1,
                "home_team": "Team 1",
                "away_team_id": 5,
                "away_team": "Team 5",
                "home_goals": 1,
                "away_goals": 1,
                "phase": "Group",
                "round": "3",
                "match_type": "Competitive",
                "result_after_90": "DRAW",
            },
        ]
    )


def test_same_timestamp_matches_use_identical_pre_timestamp_history():
    features = build_features(synthetic_matches())
    same_timestamp = features.set_index("match_id").loc[["game-a", "game-b"]]

    historical_columns = [
        "home_form_matches",
        "home_form_points",
        "home_form_ppg",
        "home_matches_played",
        "home_overall_ppg",
        "home_overall_win_rate",
        "home_overall_goals_for_pg",
        "home_overall_goals_against_pg",
        "home_team_home_matches",
        "home_team_home_ppg",
        "home_team_home_win_rate",
    ]

    assert same_timestamp.loc["game-a", historical_columns].equals(
        same_timestamp.loc["game-b", historical_columns]
    )
    assert same_timestamp.loc["game-a", "home_matches_played"] == 1
    differential_columns = [
        "ppg_diff",
        "overall_ppg_diff",
        "goal_diff_pg",
        "expected_goal_margin",
    ]
    assert same_timestamp.loc["game-a", differential_columns].equals(
        same_timestamp.loc["game-b", differential_columns]
    )
    assert same_timestamp.loc["game-a", "ppg_diff"] == 3.0
    assert same_timestamp.loc["game-a", "expected_goal_margin"] == 2.0


def test_same_timestamp_result_cannot_influence_another_match():
    features = build_features(synthetic_matches()).set_index("match_id")

    assert features.loc["game-a", "home_form_points"] == 3
    assert features.loc["game-b", "home_form_points"] == 3
    assert features.loc["game-a", "home_form_goals_for"] == 2
    assert features.loc["game-b", "home_form_goals_for"] == 2


def test_timestamp_results_update_history_for_next_timestamp():
    features = build_features(synthetic_matches()).set_index("match_id")

    assert features.loc["after-t", "home_form_matches"] == 3
    assert features.loc["after-t", "home_form_points"] == 6
    assert features.loc["after-t", "home_matches_played"] == 3
    assert features.loc["after-t", "home_overall_goals_for_pg"] == 5 / 3


def test_first_match_has_zero_initialized_historical_features():
    features = build_features(synthetic_matches()).set_index("match_id")
    first_match = features.loc["before-t"]

    zero_features = [
        "home_form_matches",
        "home_form_points",
        "home_form_ppg",
        "home_matches_played",
        "home_overall_ppg",
        "home_overall_win_rate",
        "home_overall_goals_for_pg",
        "home_overall_goals_against_pg",
        "home_team_home_matches",
        "home_team_home_ppg",
        "home_team_home_win_rate",
        "away_form_matches",
        "away_form_points",
        "away_form_ppg",
        "away_matches_played",
        "away_overall_ppg",
        "away_overall_win_rate",
        "away_overall_goals_for_pg",
        "away_overall_goals_against_pg",
        "away_team_away_matches",
        "away_team_away_ppg",
        "away_team_away_win_rate",
    ]

    assert (first_match[zero_features] == 0).all()