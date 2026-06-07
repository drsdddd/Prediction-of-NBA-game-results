from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from nba_nn.data import add_matchup_difference_features
from nba_nn.model import MarginNBAPredictor


DEFAULT_INPUT = Path("data_preparation/data/processed/features_multi_seasons.csv")
DEFAULT_ARTIFACT_DIR = Path("artifacts/multitask_nn")
DEFAULT_OUTPUT = Path("artifacts/multitask_nn/predictions.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained NBA margin model on matchup features.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to a matchup feature CSV.")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing trained model and preprocessors.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save predictions.")
    return parser.parse_args()


def margin_to_win_prob(margin: np.ndarray) -> np.ndarray:
    clipped_margin = np.clip(margin, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped_margin))


def load_model(artifact_dir: Path, device: torch.device) -> MarginNBAPredictor:
    checkpoint = torch.load(artifact_dir / "multitask_model.pt", map_location=device)
    model = MarginNBAPredictor(
        input_dim=checkpoint["input_dim"],
        num_home_teams=checkpoint["num_home_teams"],
        num_away_teams=checkpoint["num_away_teams"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    if not args.artifacts.exists():
        raise FileNotFoundError(f"Artifact directory not found: {args.artifacts}")

    feature_scaler = joblib.load(args.artifacts / "feature_scaler.joblib")
    margin_scaler = joblib.load(args.artifacts / "margin_scaler.joblib")
    feature_cols = joblib.load(args.artifacts / "feature_columns.joblib")
    category_maps = joblib.load(args.artifacts / "category_maps.joblib")

    df = pd.read_csv(args.input)
    feature_df = add_matchup_difference_features(df)

    missing_cols = [col for col in feature_cols if col not in feature_df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV is missing required feature columns: {missing_cols}")

    feature_matrix = feature_scaler.transform(feature_df[feature_cols].values)
    feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
    categorical_matrix = []
    for col in ["TEAM_ID_HOME", "TEAM_ID_AWAY"]:
        encoded = df[col].astype(int).map(category_maps[col])
        if encoded.isna().any():
            missing_values = df.loc[encoded.isna(), col].unique().tolist()
            raise ValueError(f"Unseen categorical values in {col}: {missing_values}")
        categorical_matrix.append(encoded.to_numpy(dtype=np.int64))
    categorical_tensor = torch.tensor(np.column_stack(categorical_matrix), dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.artifacts, device)

    with torch.no_grad():
        outputs = model(feature_tensor.to(device), categorical_tensor.to(device))
        margin_pred_scaled = outputs["margin_pred"].cpu().numpy().reshape(-1, 1)

    margin_pred = margin_scaler.inverse_transform(margin_pred_scaled).reshape(-1)
    win_prob = margin_to_win_prob(margin_pred)

    result_df = df.copy()
    result_df["PRED_WIN_PROB_HOME"] = win_prob
    result_df["PRED_WIN_HOME"] = (margin_pred > 0).astype(int)
    result_df["PRED_PLUS_MINUS_HOME"] = margin_pred

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)

    preview_cols = [
        col
        for col in [
            "GAME_DATE",
            "TEAM_ABBREVIATION_HOME",
            "TEAM_ABBREVIATION_AWAY",
            "PRED_WIN_PROB_HOME",
            "PRED_WIN_HOME",
            "PRED_PLUS_MINUS_HOME",
        ]
        if col in result_df.columns
    ]

    print(f"Saved predictions to: {args.output}")
    if preview_cols:
        print(result_df[preview_cols].head())


if __name__ == "__main__":
    main()
