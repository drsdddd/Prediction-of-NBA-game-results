from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


TARGET_WIN_COL = "TARGET_WIN_HOME"
TARGET_MARGIN_COL = "TARGET_PLUS_MINUS_HOME"

NON_FEATURE_COLUMNS = {
    "GAME_ID",
    "GAME_DATE",
    "GAME_DATE_AWAY",
    "TEAM_ABBREVIATION_HOME",
    "TEAM_ABBREVIATION_AWAY",
    TARGET_WIN_COL,
    TARGET_MARGIN_COL,
}

CATEGORICAL_COLUMNS = [
    "TEAM_ID_HOME",
    "TEAM_ID_AWAY",
]


@dataclass
class SplitData:
    features: np.ndarray
    categorical_features: np.ndarray
    win_target: np.ndarray
    margin_target: np.ndarray


class NBAGameDataset(Dataset):
    """PyTorch dataset wrapper for tabular NBA matchup features."""

    def __init__(self, split_data: SplitData) -> None:
        self.features = torch.tensor(split_data.features, dtype=torch.float32)
        self.categorical_features = torch.tensor(split_data.categorical_features, dtype=torch.long)
        self.win_target = torch.tensor(split_data.win_target, dtype=torch.float32).view(-1, 1)
        self.margin_target = torch.tensor(split_data.margin_target, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "categorical_features": self.categorical_features[index],
            "win_target": self.win_target[index],
            "margin_target": self.margin_target[index],
        }


def load_processed_games(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.sort_values("GAME_DATE").reset_index(drop=True)


def _paired_numeric_metrics(df: pd.DataFrame) -> List[str]:
    metrics: List[str] = []

    for col in df.columns:
        if not col.endswith("_HOME"):
            continue

        metric = col[:-5]
        away_col = f"{metric}_AWAY"
        if away_col not in df.columns:
            continue
        if metric in {"TEAM_ID", "SEASON_ID"}:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if not pd.api.types.is_numeric_dtype(df[away_col]):
            continue

        metrics.append(metric)

    return metrics


def add_matchup_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.copy()
    paired_metrics = _paired_numeric_metrics(feature_df)

    for metric in paired_metrics:
        home_col = f"{metric}_HOME"
        away_col = f"{metric}_AWAY"
        diff_col = f"DIFF_{metric}"
        abs_diff_col = f"ABS_DIFF_{metric}"
        avg_col = f"AVG_{metric}"

        feature_df[diff_col] = feature_df[home_col] - feature_df[away_col]
        feature_df[abs_diff_col] = (feature_df[home_col] - feature_df[away_col]).abs()
        feature_df[avg_col] = (feature_df[home_col] + feature_df[away_col]) / 2.0

    return feature_df


def add_season_head_to_head_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add same-season head-to-head features using only prior meetings.

    All features are computed from the current home team's perspective:
    - H2H_GAMES_PLAYED: number of earlier meetings this season
    - H2H_WIN_RATE_HOME: current home team's prior win rate in those meetings
    - H2H_AVG_MARGIN_HOME: current home team's average margin in those meetings
    - H2H_LAST_MARGIN_HOME: current home team's margin in the most recent meeting
    """
    feature_df = df.copy().sort_values("GAME_DATE").reset_index(drop=True)

    games_played = []
    win_rates = []
    avg_margins = []
    last_margins = []

    pair_history: Dict[Tuple[int, int, int], List[Tuple[int, float]]] = {}

    for _, row in feature_df.iterrows():
        season_id = int(row["SEASON_ID_HOME"])
        home_team_id = int(row["TEAM_ID_HOME"])
        away_team_id = int(row["TEAM_ID_AWAY"])
        pair_key = (season_id, min(home_team_id, away_team_id), max(home_team_id, away_team_id))

        history = pair_history.get(pair_key, [])
        current_home_margins: List[float] = []

        for prev_home_team_id, prev_margin_home in history:
            if prev_home_team_id == home_team_id:
                margin_from_current_home = prev_margin_home
            else:
                margin_from_current_home = -prev_margin_home
            current_home_margins.append(margin_from_current_home)

        games_played.append(len(current_home_margins))
        if current_home_margins:
            current_home_wins = sum(margin > 0 for margin in current_home_margins)
            win_rates.append(current_home_wins / len(current_home_margins))
            avg_margins.append(float(np.mean(current_home_margins)))
            last_margins.append(float(current_home_margins[-1]))
        else:
            win_rates.append(0.5)
            avg_margins.append(0.0)
            last_margins.append(0.0)

        pair_history.setdefault(pair_key, []).append(
            (home_team_id, float(row[TARGET_MARGIN_COL]))
        )

    feature_df["H2H_GAMES_PLAYED"] = games_played
    feature_df["H2H_WIN_RATE_HOME"] = win_rates
    feature_df["H2H_AVG_MARGIN_HOME"] = avg_margins
    feature_df["H2H_LAST_MARGIN_HOME"] = last_margins
    return feature_df


def add_pre_game_season_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add team strength features based on same-season results before the current game.

    Features are generated separately for home and away teams:
    - SEASON_WIN_PCT_PRE
    - SEASON_AVG_MARGIN_PRE
    - SEASON_GAMES_PLAYED_PRE

    Then we derive matchup gap features from them.
    """
    feature_df = df.copy().sort_values("GAME_DATE").reset_index(drop=True)

    pre_game_stats: Dict[Tuple[int, int], Dict[str, float]] = {}

    home_win_pct_pre = []
    away_win_pct_pre = []
    home_avg_margin_pre = []
    away_avg_margin_pre = []
    home_games_pre = []
    away_games_pre = []

    for _, row in feature_df.iterrows():
        season_id = int(row["SEASON_ID_HOME"])
        home_team_id = int(row["TEAM_ID_HOME"])
        away_team_id = int(row["TEAM_ID_AWAY"])

        home_key = (season_id, home_team_id)
        away_key = (season_id, away_team_id)

        home_state = pre_game_stats.get(home_key, {"games": 0, "wins": 0, "margin_sum": 0.0})
        away_state = pre_game_stats.get(away_key, {"games": 0, "wins": 0, "margin_sum": 0.0})

        home_games = int(home_state["games"])
        away_games = int(away_state["games"])

        home_games_pre.append(home_games)
        away_games_pre.append(away_games)

        home_win_pct_pre.append(home_state["wins"] / home_games if home_games > 0 else 0.5)
        away_win_pct_pre.append(away_state["wins"] / away_games if away_games > 0 else 0.5)

        home_avg_margin_pre.append(home_state["margin_sum"] / home_games if home_games > 0 else 0.0)
        away_avg_margin_pre.append(away_state["margin_sum"] / away_games if away_games > 0 else 0.0)

        current_margin = float(row[TARGET_MARGIN_COL])
        current_home_win = 1 if current_margin > 0 else 0
        current_away_win = 1 - current_home_win

        pre_game_stats[home_key] = {
            "games": home_games + 1,
            "wins": home_state["wins"] + current_home_win,
            "margin_sum": home_state["margin_sum"] + current_margin,
        }
        pre_game_stats[away_key] = {
            "games": away_games + 1,
            "wins": away_state["wins"] + current_away_win,
            "margin_sum": away_state["margin_sum"] - current_margin,
        }

    feature_df["SEASON_WIN_PCT_PRE_HOME"] = home_win_pct_pre
    feature_df["SEASON_WIN_PCT_PRE_AWAY"] = away_win_pct_pre
    feature_df["SEASON_AVG_MARGIN_PRE_HOME"] = home_avg_margin_pre
    feature_df["SEASON_AVG_MARGIN_PRE_AWAY"] = away_avg_margin_pre
    feature_df["SEASON_GAMES_PLAYED_PRE_HOME"] = home_games_pre
    feature_df["SEASON_GAMES_PLAYED_PRE_AWAY"] = away_games_pre

    feature_df["DIFF_SEASON_WIN_PCT_PRE"] = (
        feature_df["SEASON_WIN_PCT_PRE_HOME"] - feature_df["SEASON_WIN_PCT_PRE_AWAY"]
    )
    feature_df["DIFF_SEASON_AVG_MARGIN_PRE"] = (
        feature_df["SEASON_AVG_MARGIN_PRE_HOME"] - feature_df["SEASON_AVG_MARGIN_PRE_AWAY"]
    )
    feature_df["DIFF_SEASON_GAMES_PLAYED_PRE"] = (
        feature_df["SEASON_GAMES_PLAYED_PRE_HOME"] - feature_df["SEASON_GAMES_PLAYED_PRE_AWAY"]
    )
    return feature_df


def add_schedule_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rest-day and back-to-back features using only games played before the current one.

    Features:
    - REST_DAYS_HOME / AWAY
    - DIFF_REST_DAYS
    - B2B_HOME / AWAY
    """
    feature_df = df.copy().sort_values("GAME_DATE").reset_index(drop=True)

    last_game_date: Dict[Tuple[int, int], pd.Timestamp] = {}

    rest_days_home = []
    rest_days_away = []
    b2b_home = []
    b2b_away = []

    for _, row in feature_df.iterrows():
        season_id = int(row["SEASON_ID_HOME"])
        game_date = pd.Timestamp(row["GAME_DATE"])
        home_team_id = int(row["TEAM_ID_HOME"])
        away_team_id = int(row["TEAM_ID_AWAY"])

        home_key = (season_id, home_team_id)
        away_key = (season_id, away_team_id)

        prev_home_date = last_game_date.get(home_key)
        prev_away_date = last_game_date.get(away_key)

        if prev_home_date is None:
            home_rest = 7.0
        else:
            home_rest = float((game_date - prev_home_date).days - 1)
            home_rest = max(home_rest, 0.0)

        if prev_away_date is None:
            away_rest = 7.0
        else:
            away_rest = float((game_date - prev_away_date).days - 1)
            away_rest = max(away_rest, 0.0)

        rest_days_home.append(home_rest)
        rest_days_away.append(away_rest)
        b2b_home.append(1 if home_rest == 0.0 else 0)
        b2b_away.append(1 if away_rest == 0.0 else 0)

        last_game_date[home_key] = game_date
        last_game_date[away_key] = game_date

    feature_df["REST_DAYS_HOME"] = rest_days_home
    feature_df["REST_DAYS_AWAY"] = rest_days_away
    feature_df["DIFF_REST_DAYS"] = feature_df["REST_DAYS_HOME"] - feature_df["REST_DAYS_AWAY"]
    feature_df["B2B_HOME"] = b2b_home
    feature_df["B2B_AWAY"] = b2b_away
    return feature_df


def build_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feature_df = add_matchup_difference_features(df)
    feature_df = add_season_head_to_head_features(feature_df)
    feature_df = add_pre_game_season_strength_features(feature_df)
    feature_df = add_schedule_fatigue_features(feature_df)

    candidate_cols = [
        col
        for col in feature_df.columns
        if col not in NON_FEATURE_COLUMNS
        and col not in CATEGORICAL_COLUMNS
        and col not in {"SEASON_ID_HOME", "SEASON_ID_AWAY"}
    ]

    numeric_feature_cols = feature_df[candidate_cols].select_dtypes(include=["number"]).columns.tolist()
    return feature_df, numeric_feature_cols


def build_category_maps(df: pd.DataFrame) -> Dict[str, Dict[int, int]]:
    category_maps: Dict[str, Dict[int, int]] = {}

    for col in CATEGORICAL_COLUMNS:
        unique_values = sorted(df[col].dropna().astype(int).unique().tolist())
        category_maps[col] = {value: index for index, value in enumerate(unique_values)}

    return category_maps


def encode_categorical_frame(df: pd.DataFrame, category_maps: Dict[str, Dict[int, int]]) -> np.ndarray:
    encoded_columns = []

    for col in CATEGORICAL_COLUMNS:
        value_map = category_maps[col]
        encoded = df[col].astype(int).map(value_map)
        if encoded.isna().any():
            missing_values = df.loc[encoded.isna(), col].unique().tolist()
            raise ValueError(f"Unseen categorical values in {col}: {missing_values}")
        encoded_columns.append(encoded.to_numpy(dtype=np.int64))

    return np.column_stack(encoded_columns)


def seasonwise_time_split(
    df: pd.DataFrame,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for _, season_df in df.groupby("SEASON_ID_HOME", sort=True):
        season_df = season_df.sort_values("GAME_DATE").reset_index(drop=True)
        n_rows = len(season_df)
        if n_rows < 3:
            raise ValueError("Each season needs at least 3 rows for train/val/test splitting.")

        train_end = max(1, int(n_rows * train_ratio))
        val_end = max(train_end + 1, int(n_rows * (train_ratio + val_ratio)))
        val_end = min(val_end, n_rows - 1)

        train_parts.append(season_df.iloc[:train_end].copy())
        val_parts.append(season_df.iloc[train_end:val_end].copy())
        test_parts.append(season_df.iloc[val_end:].copy())

    train_df = pd.concat(train_parts, ignore_index=True).sort_values("GAME_DATE").reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sort_values("GAME_DATE").reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sort_values("GAME_DATE").reset_index(drop=True)
    return train_df, val_df, test_df


def fit_scalers(
    train_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[StandardScaler, StandardScaler]:
    feature_scaler = StandardScaler()
    margin_scaler = StandardScaler()

    feature_scaler.fit(train_df[feature_cols].values)
    margin_scaler.fit(train_df[[TARGET_MARGIN_COL]].values)
    return feature_scaler, margin_scaler


def transform_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    feature_scaler: StandardScaler,
    margin_scaler: StandardScaler,
    category_maps: Dict[str, Dict[int, int]],
) -> SplitData:
    features = feature_scaler.transform(df[feature_cols].values)
    categorical_features = encode_categorical_frame(df, category_maps)
    win_target = df[TARGET_WIN_COL].astype(np.float32).values
    margin_target = margin_scaler.transform(df[[TARGET_MARGIN_COL]].values).astype(np.float32).reshape(-1)

    return SplitData(
        features=features,
        categorical_features=categorical_features,
        win_target=win_target,
        margin_target=margin_target,
    )


def save_preprocessors(
    output_dir: str | Path,
    feature_scaler: StandardScaler,
    margin_scaler: StandardScaler,
    feature_cols: List[str],
    category_maps: Dict[str, Dict[int, int]],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(feature_scaler, output_path / "feature_scaler.joblib")
    joblib.dump(margin_scaler, output_path / "margin_scaler.joblib")
    joblib.dump(feature_cols, output_path / "feature_columns.joblib")
    joblib.dump(category_maps, output_path / "category_maps.joblib")
